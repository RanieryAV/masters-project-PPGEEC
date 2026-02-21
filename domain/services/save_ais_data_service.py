from domain.config.database_config import db
from domain.repositories.data_processing.aggregated_ais_data import AggregatedAISData
from domain.repositories.data_processing.loitering_equation_predictions import LoiteringEquationPredictions
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from pyspark.sql import functions as F
import logging
import gc

logger = logging.getLogger(__name__)

class SaveAISDataService:
    @staticmethod
    def upsert_aggregated_ais_spark_df_to_db(spark_df, batch_size: int = 1):
        expected_cols = {
            "id", "EventIndex", "trajectory", "timestamp_array", "sog_array", "cog_array",
            "behavior_type_label", "average_speed", "min_speed", "max_speed", "average_heading",
            "std_dev_heading", "total_area_time", "low_speed_percentage", "stagnation_time",
            "distance_in_kilometers", "average_time_diff_between_consecutive_points",
            "min_heading", "max_heading", "std_dev_speed", "displacement_ratio", "cog_unit_range", "cog_ratio"
        }

        missing = expected_cols - set(spark_df.columns)
        if missing:
            logger.warning(f"Spark DataFrame is missing expected columns: {missing}. Attempting to continue with available columns.")

        # Normalize/cast with Spark and filter invalid LINESTRING rows (>= 2 points)
        cleaned = spark_df.withColumn(
            "trajectory_no_wrapper",
            F.regexp_replace(F.col("trajectory"), r"^\s*LINESTRING\s*\(", "")
        ).withColumn(
            "trajectory_no_wrapper",
            F.regexp_replace(F.col("trajectory_no_wrapper"), r"\)\s*$", "")
        ).withColumn(
            "points_array",
            F.split(F.col("trajectory_no_wrapper"), r",\s*")
        )

        # unpersist later to free memory on executors
        valid_df = cleaned.filter(F.size("points_array") >= 2)
        invalid_count = cleaned.filter(F.size("points_array") < 2).count()
        if invalid_count > 0:
            logger.warning(f"Filtered out {invalid_count} invalid trajectories with fewer than 2 points.")

        # Cast textual array columns to strings in Spark (keeps them as Spark strings)
        cast_cols = ["trajectory", "timestamp_array", "sog_array", "cog_array", "behavior_type_label"]
        for c in cast_cols:
            if c in valid_df.columns:
                valid_df = valid_df.withColumn(c, F.col(c).cast("string"))

        # Prepare iterator and housekeeping
        iterator = valid_df.toLocalIterator()
        processed = 0
        inserted_or_updated = 0

        # Precompute updatable column names (don't include the primary key)
        updatable_cols = [c.name for c in AggregatedAISData.__table__.columns if c.name != "primary_key"]

        # helper that converts a Spark Row to a DB dict, using one asDict() call
        def row_to_db_dict_once(r):
            rdict = r.asDict()  # only call once
            # Map CSV id -> mmsi
            db_dict = {
                "mmsi": rdict.get("id"),
                "event_index": int(rdict["EventIndex"]) if rdict.get("EventIndex") is not None else None,
                "trajectory": rdict.get("trajectory"),
                "timestamp_array": rdict.get("timestamp_array"),
                "sog_array": rdict.get("sog_array"),
                "cog_array": rdict.get("cog_array"),
                "behavior_type_label": rdict.get("behavior_type_label"),
                "average_speed": float(rdict["average_speed"]) if rdict.get("average_speed") is not None else None,
                "min_speed": float(rdict["min_speed"]) if rdict.get("min_speed") is not None else None,
                "max_speed": float(rdict["max_speed"]) if rdict.get("max_speed") is not None else None,
                "average_heading": float(rdict["average_heading"]) if rdict.get("average_heading") is not None else None,
                "std_dev_heading": float(rdict["std_dev_heading"]) if rdict.get("std_dev_heading") is not None else None,
                "total_area_time": float(rdict["total_area_time"]) if rdict.get("total_area_time") is not None else None,
                "low_speed_percentage": float(rdict["low_speed_percentage"]) if rdict.get("low_speed_percentage") is not None else None,
                "stagnation_time": float(rdict["stagnation_time"]) if rdict.get("stagnation_time") is not None else None,
                "distance_in_kilometers": float(rdict["distance_in_kilometers"]) if rdict.get("distance_in_kilometers") is not None else None,
                "average_time_diff_between_consecutive_points": float(rdict["average_time_diff_between_consecutive_points"]) if rdict.get("average_time_diff_between_consecutive_points") is not None else None,
                "min_heading": float(rdict["min_heading"]) if rdict.get("min_heading") is not None else None,
                "max_heading": float(rdict["max_heading"]) if rdict.get("max_heading") is not None else None,
                "std_dev_speed": float(rdict["std_dev_speed"]) if rdict.get("std_dev_speed") is not None else None,
                "displacement_ratio": float(rdict["displacement_ratio"]) if rdict.get("displacement_ratio") is not None else None,
                "cog_unit_range": float(rdict["cog_unit_range"]) if rdict.get("cog_unit_range") is not None else None,
                "cog_ratio": float(rdict["cog_ratio"]) if rdict.get("cog_ratio") is not None else None,
            }
            # do not keep rdict reference
            return db_dict

        # If user requested batch_size == 1, do immediate single-row upserts (minimal driver memory)
        try:
            if batch_size == 1:
                # Single-row upsert loop
                for row in iterator:
                    processed += 1
                    db_row = row_to_db_dict_once(row)

                    try:
                        stmt = insert(AggregatedAISData).values(db_row)
                        # construct update mapping using precomputed column names
                        update_dict = {col: getattr(stmt.excluded, col) for col in updatable_cols}

                        stmt = stmt.on_conflict_do_update(
                            constraint="unique_mmsi_event_index_behavior_type_label",
                            set_=update_dict
                        )

                        db.session.execute(stmt)
                        db.session.commit()
                        inserted_or_updated += 1

                    except SQLAlchemyError as e:
                        db.session.rollback()
                        logger.exception(f"SQLAlchemy error while upserting row at processed={processed}: {e}")
                    except Exception as e:
                        db.session.rollback()
                        logger.exception(f"Unexpected error while upserting row at processed={processed}: {e}")
                    finally:
                        # remove reference to db_row and clear SQLAlchemy-managed state
                        try:
                            db.session.expunge_all()
                            db.session.close()
                        except Exception:
                            # session cleanup best-effort
                            pass
                        del db_row

                    # periodic GC to free large objects (tune frequency if needed)
                    if processed % 200 == 0:
                        gc.collect()
                        logger.debug(f"Periodic GC at processed={processed}")

            else:
                # small-batch path (keeps small list, clears immediately after commit)
                batch = []
                for row in iterator:
                    processed += 1
                    db_row = row_to_db_dict_once(row)
                    batch.append(db_row)
                    del db_row  # free right away

                    if len(batch) >= batch_size:
                        try:
                            stmt = insert(AggregatedAISData).values(batch)
                            update_dict = {col: getattr(stmt.excluded, col) for col in updatable_cols}
                            stmt = stmt.on_conflict_do_update(
                                constraint="unique_mmsi_event_index_behavior_type_label",
                                set_=update_dict
                            )
                            db.session.execute(stmt)
                            db.session.commit()
                            inserted_or_updated += len(batch)
                        except SQLAlchemyError as e:
                            db.session.rollback()
                            logger.exception(f"SQLAlchemy error while upserting batch at processed={processed}: {e}")
                        except Exception as e:
                            db.session.rollback()
                            logger.exception(f"Unexpected error while upserting batch at processed={processed}: {e}")
                        finally:
                            # free the batch contents and the list itself
                            batch.clear()
                            gc.collect()
                            try:
                                db.session.expunge_all()
                                db.session.close()
                            except Exception:
                                pass

                # final flush if anything left
                if batch:
                    try:
                        stmt = insert(AggregatedAISData).values(batch)
                        update_dict = {col: getattr(stmt.excluded, col) for col in updatable_cols}
                        stmt = stmt.on_conflict_do_update(
                            constraint="unique_mmsi_event_index_behavior_type_label",
                            set_=update_dict
                        )
                        db.session.execute(stmt)
                        db.session.commit()
                        inserted_or_updated += len(batch)
                    except SQLAlchemyError as e:
                        db.session.rollback()
                        logger.exception(f"SQLAlchemy error on final batch: {e}")
                    except Exception as e:
                        db.session.rollback()
                        logger.exception(f"Unexpected error on final batch: {e}")
                    finally:
                        batch.clear()
                        try:
                            db.session.expunge_all()
                            db.session.close()
                        except Exception:
                            pass
                        del batch
                        gc.collect()

        except Exception as e:
            logger.exception(f"Error while iterating Spark rows: {e}")
        finally:
            # ensure Spark dataframes are released from executors and free local references
            try:
                cleaned.unpersist(blocking=False)
            except Exception:
                pass
            try:
                valid_df.unpersist(blocking=False)
            except Exception:
                pass

            # close/cleanup the session fully
            try:
                db.session.remove()
            except Exception:
                try:
                    db.session.close()
                except Exception:
                    pass

            # remove references and trigger GC
            del iterator
            del cleaned
            del valid_df
            gc.collect()

            logger.info(
                f"Completed streaming. Processed {processed} valid rows; upserted {inserted_or_updated} rows. Invalid rows filtered: {invalid_count}."
            )

        return {
            "processed_rows": processed,
            "upserted_rows": inserted_or_updated,
            "invalid_rows_filtered": invalid_count
        }
    
    @staticmethod
    def upsert_agg_ais_classified_by_lotering_equation_spark_df_to_db(spark_df, batch_size: int = 1):
        expected_cols = {
            "mmsi", "event_index", "trajectory", "timestamp_array", "sog_array", "cog_array",
            "behavior_type_label", "average_speed", "min_speed", "max_speed", "average_heading",
            "std_dev_heading", "total_area_time", "low_speed_percentage", "stagnation_time",
            "distance_in_kilometers", "average_time_diff_between_consecutive_points",
            "min_heading", "max_heading", "std_dev_speed", "displacement_ratio", "cog_unit_range", "cog_ratio",
            "trajectory_redundancy", "behavior_type_by_loitering_equation" 
        }

        missing = expected_cols - set(spark_df.columns)
        if missing:
            logger.warning(f"Spark DataFrame is missing expected columns: {missing}. Attempting to continue with available columns.")

        # Normalize/cast with Spark and filter invalid LINESTRING rows (>= 2 points)
        cleaned = spark_df.withColumn(
            "trajectory_no_wrapper",
            F.regexp_replace(F.col("trajectory"), r"^\s*LINESTRING\s*\(", "")
        ).withColumn(
            "trajectory_no_wrapper",
            F.regexp_replace(F.col("trajectory_no_wrapper"), r"\)\s*$", "")
        ).withColumn(
            "points_array",
            F.split(F.col("trajectory_no_wrapper"), r",\s*")
        )

        # unpersist later to free memory on executors
        valid_df = cleaned.filter(F.size("points_array") >= 2)
        invalid_count = cleaned.filter(F.size("points_array") < 2).count()
        if invalid_count > 0:
            logger.warning(f"Filtered out {invalid_count} invalid trajectories with fewer than 2 points.")

        # Cast textual array columns to strings in Spark (keeps them as Spark strings)
        cast_cols = ["trajectory", "timestamp_array", "sog_array", "cog_array", "behavior_type_label"]
        for c in cast_cols:
            if c in valid_df.columns:
                valid_df = valid_df.withColumn(c, F.col(c).cast("string"))

        # Prepare iterator and housekeeping
        iterator = valid_df.toLocalIterator()
        processed = 0
        inserted_or_updated = 0

        # Precompute updatable column names (don't include the primary key)
        updatable_cols = [c.name for c in LoiteringEquationPredictions.__table__.columns if c.name != "primary_key"]

        # ---------- Minimal helpers added ----------
        def _to_scalar(v):
            """If v is a list/tuple, return first element (or None if empty). Else return v."""
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                return v[0] if len(v) > 0 else None
            return v

        def _to_int(v):
            s = _to_scalar(v)
            if s is None:
                return None
            try:
                return int(s)
            except Exception:
                # try float -> int (defensive), otherwise None
                try:
                    return int(float(s))
                except Exception:
                    return None

        def _to_float(v):
            s = _to_scalar(v)
            if s is None:
                return None
            try:
                return float(s)
            except Exception:
                return None

        def _to_str_scalar(v):
            """
            Convert v to a plain string scalar:
            - if v is list/tuple -> take first element (and convert to str)
            - if v is a string that looks like "['X']" or '["X","Y"]' -> extract first item without brackets/quotes
            - otherwise return str(v)
            Returns None if input is None or empty.
            """
            if v is None:
                return None
            # list/tuple -> take first element
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    return None
                first = v[0]
                return None if first is None else str(first)
            s = str(v).strip()
            if s == "":
                return None
            # if stringified list like "['LOITERING']" or '["LOITERING","X"]'
            if s.startswith("[") and s.endswith("]"):
                inner = s[1:-1].strip()
                if inner == "":
                    return None
                # take first element before any comma
                first_part = inner.split(",", 1)[0].strip()
                # remove surrounding quotes if present
                if (first_part.startswith("'") and first_part.endswith("'")) or (first_part.startswith('"') and first_part.endswith('"')):
                    first_part = first_part[1:-1]
                return first_part
            # remove surrounding single/double quotes if present
            if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
                return s[1:-1]
            return s
        # -------------------------------------------

        # helper that converts a Spark Row to a DB dict, using one asDict() call
        def row_to_db_dict_once(r):
            rdict = r.asDict()  # only call once

            # Map array-like primary_key -> single integer aggregated_ais_data_id (take first element)
            aggregated_pk = _to_int(rdict.get("primary_key_from_aggregated_ais_data"))

            # Map event_index (may be list/array) -> single integer (take first element)
            event_index_scalar = _to_int(rdict.get("event_index"))

            # Use scalar conversions for numeric fields that might be passed as arrays
            traj_redundancy_scalar = _to_float(rdict.get("trajectory_redundancy"))
            avg_speed = _to_float(rdict.get("average_speed"))
            min_sp = _to_float(rdict.get("min_speed"))
            max_sp = _to_float(rdict.get("max_speed"))
            avg_heading = _to_float(rdict.get("average_heading"))
            std_h = _to_float(rdict.get("std_dev_heading"))
            total_area = _to_float(rdict.get("total_area_time"))
            low_speed_pct = _to_float(rdict.get("low_speed_percentage"))
            stagn_time = _to_float(rdict.get("stagnation_time"))
            dist_km = _to_float(rdict.get("distance_in_kilometers"))
            avg_time_diff = _to_float(rdict.get("average_time_diff_between_consecutive_points"))
            min_h = _to_float(rdict.get("min_heading"))
            max_h = _to_float(rdict.get("max_heading"))
            std_s = _to_float(rdict.get("std_dev_speed"))
            disp_ratio = _to_float(rdict.get("displacement_ratio"))
            cog_range = _to_float(rdict.get("cog_unit_range"))
            cog_r = _to_float(rdict.get("cog_ratio"))

            # Extract a clean string scalar for behavior_type_label (removes [ ] and quotes if present)
            behavior_label_scalar = _to_str_scalar(rdict.get("behavior_type_label"))

            # Build db_dict using scalars (preserve textual arrays as strings where appropriate)
            db_dict = {
                "aggregated_ais_data_id": aggregated_pk,
                "mmsi": rdict.get("mmsi"),
                "event_index": event_index_scalar,
                "trajectory": rdict.get("trajectory"),
                "timestamp_array": rdict.get("timestamp_array"),
                "sog_array": rdict.get("sog_array"),
                "cog_array": rdict.get("cog_array"),
                "behavior_type_label": behavior_label_scalar,
                "behavior_type_by_loitering_equation": rdict.get("behavior_type_by_loitering_equation"),
                "trajectory_redundancy": traj_redundancy_scalar,
                "average_speed": avg_speed,
                "min_speed": min_sp,
                "max_speed": max_sp,
                "average_heading": avg_heading,
                "std_dev_heading": std_h,
                "total_area_time": total_area,
                "low_speed_percentage": low_speed_pct,
                "stagnation_time": stagn_time,
                "distance_in_kilometers": dist_km,
                "average_time_diff_between_consecutive_points": avg_time_diff,
                "min_heading": min_h,
                "max_heading": max_h,
                "std_dev_speed": std_s,
                "displacement_ratio": disp_ratio,
                "cog_unit_range": cog_range,
                "cog_ratio": cog_r,
            }
            # do not keep rdict reference
            return db_dict

        # If user requested batch_size == 1, do immediate single-row upserts (minimal driver memory)
        try:
            if batch_size == 1:
                # Single-row upsert loop
                for row in iterator:
                    processed += 1
                    db_row = row_to_db_dict_once(row)

                    try:
                        stmt = insert(LoiteringEquationPredictions).values(db_row)
                        # construct update mapping using precomputed column names
                        update_dict = {col: getattr(stmt.excluded, col) for col in updatable_cols}

                        stmt = stmt.on_conflict_do_update(
                            constraint="unique_mmsi_event_index_behavior_type_label_timestamp_array",
                            set_=update_dict
                        )

                        db.session.execute(stmt)
                        db.session.commit()
                        inserted_or_updated += 1

                    except SQLAlchemyError as e:
                        db.session.rollback()
                        logger.exception(f"SQLAlchemy error while upserting row at processed={processed}: {e}")
                    except Exception as e:
                        db.session.rollback()
                        logger.exception(f"Unexpected error while upserting row at processed={processed}: {e}")
                    finally:
                        # remove reference to db_row and clear SQLAlchemy-managed state
                        try:
                            db.session.expunge_all()
                            db.session.close()
                        except Exception:
                            # session cleanup best-effort
                            pass
                        del db_row

                    # periodic GC to free large objects (tune frequency if needed)
                    if processed % 200 == 0:
                        gc.collect()
                        logger.debug(f"Periodic GC at processed={processed}")

            else:
                # small-batch path (keeps small list, clears immediately after commit)
                batch = []
                for row in iterator:
                    processed += 1
                    db_row = row_to_db_dict_once(row)
                    batch.append(db_row)
                    del db_row  # free right away

                    if len(batch) >= batch_size:
                        try:
                            stmt = insert(LoiteringEquationPredictions).values(batch)
                            update_dict = {col: getattr(stmt.excluded, col) for col in updatable_cols}
                            stmt = stmt.on_conflict_do_update(
                                constraint="unique_mmsi_event_index_behavior_type_label_timestamp_array",
                                set_=update_dict
                            )
                            db.session.execute(stmt)
                            db.session.commit()
                            inserted_or_updated += len(batch)
                        except SQLAlchemyError as e:
                            db.session.rollback()
                            logger.exception(f"SQLAlchemy error while upserting batch at processed={processed}: {e}")
                        except Exception as e:
                            db.session.rollback()
                            logger.exception(f"Unexpected error while upserting batch at processed={processed}: {e}")
                        finally:
                            # free the batch contents and the list itself
                            batch.clear()
                            gc.collect()
                            try:
                                db.session.expunge_all()
                                db.session.close()
                            except Exception:
                                pass

                # final flush if anything left
                if batch:
                    try:
                        stmt = insert(LoiteringEquationPredictions).values(batch)
                        update_dict = {col: getattr(stmt.excluded, col) for col in updatable_cols}
                        stmt = stmt.on_conflict_do_update(
                            constraint="unique_mmsi_event_index_behavior_type_label_timestamp_array",
                            set_=update_dict
                        )
                        db.session.execute(stmt)
                        db.session.commit()
                        inserted_or_updated += len(batch)
                    except SQLAlchemyError as e:
                        db.session.rollback()
                        logger.exception(f"SQLAlchemy error on final batch: {e}")
                    except Exception as e:
                        db.session.rollback()
                        logger.exception(f"Unexpected error on final batch: {e}")
                    finally:
                        batch.clear()
                        try:
                            db.session.expunge_all()
                            db.session.close()
                        except Exception:
                            pass
                        del batch
                        gc.collect()

        except Exception as e:
            logger.exception(f"Error while iterating Spark rows: {e}")
        finally:
            # ensure Spark dataframes are released from executors and free local references
            try:
                cleaned.unpersist(blocking=False)
            except Exception:
                pass
            try:
                valid_df.unpersist(blocking=False)
            except Exception:
                pass

            # close/cleanup the session fully
            try:
                db.session.remove()
            except Exception:
                try:
                    db.session.close()
                except Exception:
                    pass

            # remove references and trigger GC
            del iterator
            del cleaned
            del valid_df
            gc.collect()

            logger.info(
                f"Completed streaming. Processed {processed} valid rows; upserted {inserted_or_updated} rows. Invalid rows filtered: {invalid_count}."
            )

        return {
            "processed_rows": processed,
            "upserted_rows": inserted_or_updated,
            "invalid_rows_filtered": invalid_count
        }

