import torch


class NodeFeatures:
    """
    Creates the node feature matrix for the
    Graph Attention Network (GAT).

    New Delhi dataset
    ------------------

    Each row represents one monitoring station.

    Output
    ------
    x : torch.FloatTensor

        Shape:

            [num_stations, num_features]

    For the new dataset:

        num_stations = 6
        num_features = 8
    """


    # ==========================================================
    # BUILD NODE FEATURES
    # ==========================================================

    @staticmethod
    def build_node_features(
        stations
    ):
        """
        Constructs the node feature matrix.

        Expected station dictionary:

            {
                "Station_ID": ...,
                "Station": ...,
                "Latitude": ...,
                "Longitude": ...,

                "Wind_Speed": ...,

                "Temperature": ...,
                "Humidity": ...,
                "Pressure": ...,

                "PM2.5": ...,
                "PM10": ...,
                "CO": ...,
                "NO2": ...
            }

        Returns
        -------
        x : torch.FloatTensor

            Shape:

                [num_stations, 8]
        """


        # ------------------------------------------------------
        # Validate input
        # ------------------------------------------------------

        if not isinstance(
            stations,
            (list, tuple)
        ):

            raise TypeError(
                "stations must be a list or tuple."
            )


        if len(stations) == 0:

            raise ValueError(
                "stations cannot be empty."
            )


        # ------------------------------------------------------
        # Feature matrix
        # ------------------------------------------------------

        feature_matrix = []


        for station in stations:

            # --------------------------------------------------
            # Required fields
            # --------------------------------------------------

            required_fields = [

                "PM2.5",

                "PM10",

                "CO",

                "NO2",

                "Temperature",

                "Humidity",

                "Pressure",

                "Wind_Speed"
            ]


            missing_fields = [

                field

                for field in required_fields

                if field not in station
            ]


            if missing_fields:

                raise KeyError(

                    "Station record is missing "
                    "required fields: "
                    +
                    ", ".join(
                        missing_fields
                    )
                )


            # --------------------------------------------------
            # Node features
            # --------------------------------------------------
            #
            # IMPORTANT:
            #
            # Keep this order fixed.
            #
            # 0 -> PM2.5
            # 1 -> PM10
            # 2 -> CO
            # 3 -> NO2
            # 4 -> Temperature
            # 5 -> Humidity
            # 6 -> Pressure
            # 7 -> Wind Speed
            #
            # These correspond to the FEATURE_COLUMNS
            # defined in preprocessing.py.
            # --------------------------------------------------

            features = [

                float(
                    station["PM2.5"]
                ),

                float(
                    station["PM10"]
                ),

                float(
                    station["CO"]
                ),

                float(
                    station["NO2"]
                ),

                float(
                    station["Temperature"]
                ),

                float(
                    station["Humidity"]
                ),

                float(
                    station["Pressure"]
                ),

                float(
                    station["Wind_Speed"]
                )
            ]


            feature_matrix.append(
                features
            )


        # ------------------------------------------------------
        # Convert to tensor
        # ------------------------------------------------------

        x = torch.tensor(

            feature_matrix,

            dtype=torch.float32
        )


        # ------------------------------------------------------
        # Numerical validation
        # ------------------------------------------------------

        if torch.isnan(x).any():

            raise ValueError(
                "Node feature matrix contains NaN values."
            )


        if torch.isinf(x).any():

            raise ValueError(
                "Node feature matrix contains Inf values."
            )


        # ------------------------------------------------------
        # Shape validation
        # ------------------------------------------------------

        expected_features = 8


        if x.shape[1] != expected_features:

            raise ValueError(

                f"Expected {expected_features} "
                f"node features, "
                f"but got {x.shape[1]}."
            )


        return x