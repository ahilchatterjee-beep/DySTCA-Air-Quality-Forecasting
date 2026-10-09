import math
import torch


class NodeFeatures:
    """
    Creates the node feature matrix for the Graph
    Attention Network (GAT).

    Each row represents one monitoring station.

    Output
    ------
    x : torch.FloatTensor
        Shape -> [num_stations, num_features]
    """
    
    # ---------------------------------------------------------
    # Wind Speed
    # ---------------------------------------------------------

    @staticmethod
    def compute_wind_speed(
        wind_u,
        wind_v
    ):
        """
        Computes wind speed from the
        horizontal wind components.
        """

        return math.sqrt(
            wind_u ** 2 +
            wind_v ** 2
        )
        
        # ---------------------------------------------------------
    # Node Feature Matrix
    # ---------------------------------------------------------

    def build_node_features(
        self,
        stations
    ):
        """
        Constructs the node feature matrix.

        Parameters
        ----------
        stations : list

        Returns
        -------
        x : torch.FloatTensor
        """

        feature_matrix = []

        for station in stations:

            wind_speed = self.compute_wind_speed(
                station["Wind_U"],
                station["Wind_V"]
            )

            features = [

                # Air Quality
                station["PM2.5 (µg/m³)"],
                station["PM10 (µg/m³)"],
                station["NO (µg/m³)"],
                station["NO2 (µg/m³)"],
                station["SO2 (µg/m³)"],
                station["CO (mg/m³)"],
                station["Ozone (µg/m³)"],

                # Meteorology
                station["Temperature"],
                station["Dew_Point"],
                wind_speed,
                station["Surface_Pressure"],
                station["Boundary_Layer_Height"],
                station["Rainfall"],
                station["TOT-RF (mm)"]

            ]

            feature_matrix.append(
                features
            )

        x = torch.tensor(
            feature_matrix,
            dtype=torch.float32
        )

        return x