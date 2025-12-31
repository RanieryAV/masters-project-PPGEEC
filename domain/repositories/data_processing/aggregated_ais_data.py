from domain.config.database_config import db
from geoalchemy2 import Geometry
from geoalchemy2.shape import to_shape
from sqlalchemy import UniqueConstraint, ARRAY

class AggregatedAISData(db.Model):
    __tablename__ = 'aggregated_ais_data'
    # schema only; no extra columns beyond those requested
    __table_args__ = (
        UniqueConstraint('mmsi', 'event_index', 'behavior_type_label', name='unique_mmsi_event_index_behavior_type_label'),
        {"schema": "captaima"},
    )

    primary_key = db.Column(db.Integer, primary_key=True, autoincrement=True)
    mmsi = db.Column('mmsi', db.String(255), nullable=False)
    event_index = db.Column('event_index', db.Integer, nullable=False)
    trajectory = db.Column(Geometry(geometry_type='LINESTRING', srid=4326), nullable=False)
    timestamp_array = db.Column('timestamp_array', db.Text, nullable=False)
    sog_array = db.Column('sog_array', db.Text, nullable=False)   # speed array
    cog_array = db.Column('cog_array', db.Text, nullable=False)   # heading array
    behavior_type_label = db.Column('behavior_type_label', db.String(255), nullable=False)

    average_speed = db.Column('average_speed', db.Float, nullable=False)
    min_speed = db.Column('min_speed', db.Float, nullable=False)
    max_speed = db.Column('max_speed', db.Float, nullable=False)

    average_heading = db.Column('average_heading', db.Float, nullable=False)
    min_heading = db.Column('min_heading', db.Float, nullable=False)
    max_heading = db.Column('max_heading', db.Float, nullable=False)
    std_dev_heading = db.Column('std_dev_heading', db.Float, nullable=False)

    std_dev_speed = db.Column('std_dev_speed', db.Float, nullable=False)

    total_area_time = db.Column('total_area_time', db.Float, nullable=False)
    low_speed_percentage = db.Column('low_speed_percentage', db.Float, nullable=False)
    stagnation_time = db.Column('stagnation_time', db.Float, nullable=False)

    distance_in_kilometers = db.Column('distance_in_kilometers', db.Float, nullable=False)
    average_time_diff = db.Column('average_time_diff_between_consecutive_points', db.Float, nullable=False)

    displacement_ratio = db.Column('displacement_ratio', db.Float, nullable=False)
    cog_unit_range = db.Column('cog_unit_range', db.Float, nullable=False)
    cog_ratio = db.Column('cog_ratio', db.Float, nullable=False)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        # Convert geometry to WKT format
        trajectory_shape = to_shape(self.trajectory) if self.trajectory is not None else None
        trajectory_wkt = trajectory_shape.wkt if trajectory_shape else None

        return {
            'primary_key': self.primary_key,
            'mmsi': self.mmsi,
            'event_index': self.event_index,
            'trajectory': trajectory_wkt,
            'timestamp_array': self.timestamp_array,
            'sog_array': self.sog_array,
            'cog_array': self.cog_array,
            'behavior_type_label': self.behavior_type_label,

            'average_speed': self.average_speed,
            'min_speed': self.min_speed,
            'max_speed': self.max_speed,

            'average_heading': self.average_heading,
            'min_heading': self.min_heading,
            'max_heading': self.max_heading,
            'std_dev_heading': self.std_dev_heading,

            'std_dev_speed': self.std_dev_speed,

            'total_area_time': self.total_area_time,
            'low_speed_percentage': self.low_speed_percentage,
            'stagnation_time': self.stagnation_time,

            'distance_in_kilometers': self.distance_in_kilometers,
            'average_time_diff_between_consecutive_points': self.average_time_diff,

            'displacement_ratio': self.displacement_ratio,
            'cog_unit_range': self.cog_unit_range,
            'cog_ratio': self.cog_ratio
        }
