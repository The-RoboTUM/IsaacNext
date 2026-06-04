import yaml
from attr import dataclass


@dataclass(kw_only=True)
class ForrestParameterConfig:
    # Tendon Parameters
    gst_stiffness: float = 2 * (10 ** 5)
    dft_stiffness: float = 5 * (10 ** 4)  # FIXME: find out real value
    edt1_stiffness: float = 5 * (10 ** 5)  # FIXME: find out real value
    edt2_stiffness: float = 5 * (10 ** 5)  # FIXME: find out real value
    kft_stiffness: float = 5 * (10 ** 5)  # FIXME: find out real value

    gst_spring_rest_length: float = 0.06
    upper_gst_length: float = 0.6917  # FIXME: measure correct value
    lower_gst_length: float = 0.6314  # FIXME: measure correct value
    dft_length: float = 0.384  # FIXME: measure correct value
    edt1_length: float = 0.54  # FIXME: measure correct value
    edt2_length: float = 0.65  # FIXME: measure correct value
    kft_length: float = 0.452  # FIXME: measure correct value

    gst_damping: float = 2.0
    dft_damping: float = 2.0
    kft_damping: float = 2.0
    edt1_damping: float = 2.0
    edt2_damping: float = 2.0

    # Actuator Parameters
    actuator_stiffness = 500
    actuator_damping = 0.1
    flexor_angle = 0

    @classmethod
    def from_yaml(cls, parameter_yaml_path: str):
        """Load config from YAML file,

       Args:
           parameter_yaml_path: Path to the YAML configuration file

       Returns:
           ForrestParameterConfig: Parameters as dataclass instance
       """
        with open(parameter_yaml_path, 'r') as f:
            parameters = yaml.safe_load(f) or {}

        return cls(**parameters)
