import math
import torch


class GraphBuilder:
    """
    Builds a directed spatial graph for the proposed GAT model.

    The graph is constructed using the proposed spatial similarity equation:

        Sij = Vi * max(0, cos(thetaij)) * exp(-k * dij)

    where

        Vi      : Wind speed at source station i
        thetaij : Wind alignment angle between station i and station j
        dij     : Haversine distance between stations
        k       : Distance decay constant (optimized)

    A directed edge i -> j is created only if

        Sij >= similarity_threshold

    Self-loops are ALWAYS added after the dynamic spatial edges
    are constructed.

    Important:
        - Hyper-heuristic values of k and similarity_threshold
          are preserved.
        - Self-loops do NOT depend on the similarity threshold.
        - Self-loops guarantee that every station can retain
          its own information during GAT aggregation.
    """

    def __init__(
        self,
        decay_constant=0.5,
        similarity_threshold=0.2
    ):

        self.k = decay_constant
        self.similarity_threshold = similarity_threshold

    # ---------------------------------------------------------
    # Haversine Distance
    # ---------------------------------------------------------

    @staticmethod
    def haversine_distance(
        lat1,
        lon1,
        lat2,
        lon2
    ):
        """
        Computes the great-circle distance (in kilometers)
        between two monitoring stations.
        """

        R = 6371.0

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)

        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1)
            * math.cos(lat2)
            * math.sin(dlon / 2) ** 2
        )

        c = 2 * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )

        return R * c

    # ---------------------------------------------------------
    # Bearing Angle
    # ---------------------------------------------------------

    @staticmethod
    def calculate_bearing(
        lat1,
        lon1,
        lat2,
        lon2
    ):
        """
        Computes the bearing angle (degrees)
        from station i to station j.
        """

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)

        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        dlon = lon2 - lon1

        x = (
            math.sin(dlon)
            * math.cos(lat2)
        )

        y = (
            math.cos(lat1)
            * math.sin(lat2)
            - math.sin(lat1)
            * math.cos(lat2)
            * math.cos(dlon)
        )

        bearing = math.degrees(
            math.atan2(x, y)
        )

        return (bearing + 360) % 360

    # ---------------------------------------------------------
    # Wind Alignment Angle
    # ---------------------------------------------------------

    @staticmethod
    def compute_theta(
        wind_direction,
        bearing_angle
    ):
        """
        Computes the alignment angle between
        wind direction and station bearing.
        """

        theta = (
            wind_direction
            - bearing_angle
        )

        return math.radians(theta)

    # ---------------------------------------------------------
    # Proposed Spatial Similarity Equation
    # ---------------------------------------------------------

    def compute_similarity(
        self,
        station_i,
        station_j
    ):
        """
        Computes the proposed spatial similarity score.

        Sij = Vi × max(0, cos(thetaij)) × exp(-k × dij)
        """

        # -----------------------------------------------------
        # Extract station information
        # -----------------------------------------------------

        lat_i = station_i["Latitude"]
        lon_i = station_i["Longitude"]

        lat_j = station_j["Latitude"]
        lon_j = station_j["Longitude"]

        wind_u = station_i["Wind_U"]
        wind_v = station_i["Wind_V"]

        # -----------------------------------------------------
        # Wind Speed (Vi)
        # -----------------------------------------------------

        Vi = math.sqrt(
            wind_u ** 2
            + wind_v ** 2
        )

        # -----------------------------------------------------
        # Wind Direction
        # -----------------------------------------------------

        wind_direction = (
            math.degrees(
                math.atan2(
                    wind_u,
                    wind_v
                )
            )
            + 360
        ) % 360

        # -----------------------------------------------------
        # Distance between stations
        # -----------------------------------------------------

        dij = self.haversine_distance(
            lat_i,
            lon_i,
            lat_j,
            lon_j
        )

        # -----------------------------------------------------
        # Bearing from station i -> station j
        # -----------------------------------------------------

        bearing = self.calculate_bearing(
            lat_i,
            lon_i,
            lat_j,
            lon_j
        )

        # -----------------------------------------------------
        # Wind Alignment Angle
        # -----------------------------------------------------

        thetaij = self.compute_theta(
            wind_direction,
            bearing
        )

        # -----------------------------------------------------
        # Proposed Similarity Equation
        # -----------------------------------------------------

        Sij = (
            Vi
            * max(
                0.0,
                math.cos(thetaij)
            )
            * math.exp(
                -self.k * dij
            )
        )

        return Sij

    # ---------------------------------------------------------
    # Add Self-Loops
    # ---------------------------------------------------------

    @staticmethod
    def add_self_loops(
        edge_index,
        num_nodes
    ):
        """
        Guarantees one self-loop for every node.

        For N nodes, this adds:

            0 -> 0
            1 -> 1
            ...
            N-1 -> N-1

        Existing self-loops are not duplicated.
        """

        # -----------------------------------------------------
        # Create all required self-loops
        # -----------------------------------------------------

        self_loop_nodes = torch.arange(
            num_nodes,
            dtype=torch.long
        )

        self_loops = torch.stack(
            [
                self_loop_nodes,
                self_loop_nodes
            ],
            dim=0
        )

        # -----------------------------------------------------
        # Handle empty graph
        # -----------------------------------------------------

        if edge_index.numel() == 0:

            return self_loops

        # -----------------------------------------------------
        # Remove any existing self-loops from the original
        # graph so that we can add exactly one per node.
        # -----------------------------------------------------

        non_self_loop_mask = (
            edge_index[0]
            != edge_index[1]
        )

        edge_index = edge_index[
            :,
            non_self_loop_mask
        ]

        # -----------------------------------------------------
        # Combine spatial edges + self-loops
        # -----------------------------------------------------

        edge_index = torch.cat(
            [
                edge_index,
                self_loops
            ],
            dim=1
        )

        return edge_index

    # ---------------------------------------------------------
    # Dynamic Graph Update
    # ---------------------------------------------------------

    def update_graph(
        self,
        previous_edge_index,
        current_stations
    ):
        """
        Updates the existing graph using the latest
        meteorological conditions.

        The station locations remain fixed while only
        the directed spatial edges are updated according
        to the latest wind conditions.

        Self-loops are always added after the dynamic
        spatial edges are calculated.
        """

        updated_edge_list = []

        num_nodes = len(
            current_stations
        )

        # -----------------------------------------------------
        # Construct dynamic directed spatial edges
        # -----------------------------------------------------

        for i in range(num_nodes):

            for j in range(num_nodes):

                # Do not calculate similarity for self
                # connections. They are added separately.
                if i == j:
                    continue

                station_i = current_stations[i]
                station_j = current_stations[j]

                Sij = self.compute_similarity(
                    station_i,
                    station_j
                )

                if Sij >= self.similarity_threshold:

                    updated_edge_list.append(
                        [i, j]
                    )

        # -----------------------------------------------------
        # Convert spatial edges to tensor
        # -----------------------------------------------------

        if len(updated_edge_list) > 0:

            updated_edge_index = torch.tensor(
                updated_edge_list,
                dtype=torch.long
            ).t().contiguous()

        else:

            updated_edge_index = torch.empty(
                (2, 0),
                dtype=torch.long
            )

        # -----------------------------------------------------
        # ADD SELF-LOOPS
        # -----------------------------------------------------

        updated_edge_index = self.add_self_loops(
            updated_edge_index,
            num_nodes
        )

        return updated_edge_index

    # ---------------------------------------------------------
    # Initial Graph Construction
    # ---------------------------------------------------------

    def build_graph(
        self,
        stations
    ):
        """
        Constructs the initial directed spatial graph.

        This method is called once at the beginning
        of the forecasting process.

        Self-loops are guaranteed for every station.
        """

        edge_list = []

        num_nodes = len(stations)

        # -----------------------------------------------------
        # Construct dynamic spatial edges
        # -----------------------------------------------------

        for i in range(num_nodes):

            for j in range(num_nodes):

                # Self-loops are handled separately.
                if i == j:
                    continue

                station_i = stations[i]
                station_j = stations[j]

                Sij = self.compute_similarity(
                    station_i,
                    station_j
                )

                if Sij >= self.similarity_threshold:

                    edge_list.append(
                        [i, j]
                    )

        # -----------------------------------------------------
        # Convert spatial edges to tensor
        # -----------------------------------------------------

        if len(edge_list) > 0:

            edge_index = torch.tensor(
                edge_list,
                dtype=torch.long
            ).t().contiguous()

        else:

            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long
            )

        # -----------------------------------------------------
        # ADD SELF-LOOPS
        # -----------------------------------------------------

        edge_index = self.add_self_loops(
            edge_index,
            num_nodes
        )

        return edge_index