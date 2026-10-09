# ==========================================================
# graph_builder.py
#
# WIND-SPEED + DISTANCE DYNAMIC GRAPH
#
# For the new Delhi dataset.
#
# Spatial similarity:
#
#     Sij = Vi * exp(-k * dij)
#
# where:
#
#     Vi = wind speed at source station i
#     dij = Haversine distance between station i and j
#     k = optimized distance decay constant
#
# A directed edge i -> j is created when:
#
#     Sij >= similarity_threshold
#
# Self-loops are ALWAYS added.
# ==========================================================


import math
import torch


class GraphBuilder:

    """
    Builds a directed dynamic spatial graph.

    The graph is constructed using:

        Sij = Vi * exp(-k * dij)

    where:

        Vi
            Wind speed at source station i.

        dij
            Haversine distance between station i
            and station j in kilometers.

        k
            Distance decay constant.

    The new Delhi dataset provides wind speed but
    does not provide wind direction.

    Therefore this implementation intentionally
    does NOT fabricate Wind_U / Wind_V.

    A directed edge i -> j is created only when:

        Sij >= similarity_threshold

    Self-loops are always added.
    """

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(
        self,
        decay_constant=0.5,
        similarity_threshold=0.2
    ):

        self.k = float(
            decay_constant
        )

        self.similarity_threshold = float(
            similarity_threshold
        )


    # ======================================================
    # HAVERSINE DISTANCE
    # ======================================================

    @staticmethod
    def haversine_distance(
        lat1,
        lon1,
        lat2,
        lon2
    ):

        """
        Computes great-circle distance in kilometers.
        """

        R = 6371.0

        lat1 = math.radians(
            lat1
        )

        lon1 = math.radians(
            lon1
        )

        lat2 = math.radians(
            lat2
        )

        lon2 = math.radians(
            lon2
        )

        dlat = (
            lat2 - lat1
        )

        dlon = (
            lon2 - lon1
        )

        a = (

            math.sin(
                dlat / 2
            ) ** 2

            +

            math.cos(lat1)

            * math.cos(lat2)

            * math.sin(
                dlon / 2
            ) ** 2
        )

        c = 2 * math.atan2(

            math.sqrt(a),

            math.sqrt(
                1 - a
            )
        )

        return R * c


    # ======================================================
    # EXTRACT WIND SPEED
    # ======================================================

    @staticmethod
    def get_wind_speed(
        station
    ):

        """
        Extracts wind speed from the station record.

        Preferred field:

            Wind_Speed

        Compatibility:

            windspeed_kph

        Optional compatibility with old data:

            Wind_U / Wind_V
        """

        # --------------------------------------------------
        # New Delhi dataset
        # --------------------------------------------------

        if "Wind_Speed" in station:

            return float(
                station["Wind_Speed"]
            )


        if "windspeed_kph" in station:

            return float(
                station["windspeed_kph"]
            )


        # --------------------------------------------------
        # Backward compatibility
        # --------------------------------------------------
        #
        # If an older dataset still contains Wind_U/V,
        # we can calculate speed.
        # --------------------------------------------------

        if (
            "Wind_U" in station

            and

            "Wind_V" in station
        ):

            wind_u = float(
                station["Wind_U"]
            )

            wind_v = float(
                station["Wind_V"]
            )

            return math.sqrt(

                wind_u ** 2

                +

                wind_v ** 2
            )


        raise KeyError(

            "Station record does not contain "
            "wind-speed information.\n\n"

            "Expected one of:\n"

            "    Wind_Speed\n"
            "    windspeed_kph\n"
            "    Wind_U + Wind_V"
        )


    # ======================================================
    # SPATIAL SIMILARITY
    # ======================================================

    def compute_similarity(
        self,
        station_i,
        station_j
    ):

        """
        Computes:

            Sij = Vi * exp(-k * dij)

        This version intentionally excludes wind
        direction because the new Delhi dataset
        does not provide it.
        """

        # --------------------------------------------------
        # Coordinates
        # --------------------------------------------------

        lat_i = float(
            station_i["Latitude"]
        )

        lon_i = float(
            station_i["Longitude"]
        )

        lat_j = float(
            station_j["Latitude"]
        )

        lon_j = float(
            station_j["Longitude"]
        )


        # --------------------------------------------------
        # Wind speed at source station
        # --------------------------------------------------

        Vi = self.get_wind_speed(
            station_i
        )


        # --------------------------------------------------
        # Distance
        # --------------------------------------------------

        dij = self.haversine_distance(

            lat_i,
            lon_i,

            lat_j,
            lon_j
        )


        # --------------------------------------------------
        # Distance-decay term
        # --------------------------------------------------

        distance_decay = math.exp(

            -self.k * dij
        )


        # --------------------------------------------------
        # Final similarity
        # --------------------------------------------------

        Sij = (

            Vi

            *

            distance_decay
        )


        return Sij


    # ======================================================
    # ADD SELF LOOPS
    # ======================================================

    @staticmethod
    def add_self_loops(
        edge_index,
        num_nodes
    ):

        """
        Guarantees exactly one self-loop
        for every station.
        """

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


        # --------------------------------------------------
        # Empty graph
        # --------------------------------------------------

        if edge_index.numel() == 0:

            return self_loops


        # --------------------------------------------------
        # Remove existing self-loops
        # --------------------------------------------------

        non_self_loop_mask = (

            edge_index[0]

            !=

            edge_index[1]
        )


        edge_index = edge_index[
            :,
            non_self_loop_mask
        ]


        # --------------------------------------------------
        # Add exactly one self-loop per station
        # --------------------------------------------------

        edge_index = torch.cat(

            [
                edge_index,
                self_loops
            ],

            dim=1
        )


        return edge_index


    # ======================================================
    # DYNAMIC GRAPH UPDATE
    # ======================================================

    def update_graph(
        self,
        previous_edge_index,
        current_stations
    ):

        """
        Updates the graph using the latest station
        wind-speed conditions.

        Coordinates remain fixed.

        Wind speed can change with time.

        Therefore the graph can evolve over time.
        """

        updated_edge_list = []


        num_nodes = len(
            current_stations
        )


        # --------------------------------------------------
        # Dynamic directed edges
        # --------------------------------------------------

        for i in range(
            num_nodes
        ):

            for j in range(
                num_nodes
            ):

                # Self-loops are added separately.
                if i == j:

                    continue


                station_i = (
                    current_stations[i]
                )

                station_j = (
                    current_stations[j]
                )


                Sij = self.compute_similarity(

                    station_i,

                    station_j
                )


                if (
                    Sij
                    >=
                    self.similarity_threshold
                ):

                    updated_edge_list.append(

                        [
                            i,
                            j
                        ]
                    )


        # --------------------------------------------------
        # Convert to edge_index
        # --------------------------------------------------

        if len(
            updated_edge_list
        ) > 0:

            updated_edge_index = torch.tensor(

                updated_edge_list,

                dtype=torch.long
            ).t().contiguous()


        else:

            updated_edge_index = torch.empty(

                (2, 0),

                dtype=torch.long
            )


        # --------------------------------------------------
        # Add self-loops
        # --------------------------------------------------

        updated_edge_index = (
            self.add_self_loops(

                updated_edge_index,

                num_nodes
            )
        )


        return updated_edge_index


    # ======================================================
    # INITIAL GRAPH CONSTRUCTION
    # ======================================================

    def build_graph(
        self,
        stations
    ):

        """
        Constructs the initial graph from the first
        timestep's station conditions.
        """

        edge_list = []


        num_nodes = len(
            stations
        )


        # --------------------------------------------------
        # Construct spatial edges
        # --------------------------------------------------

        for i in range(
            num_nodes
        ):

            for j in range(
                num_nodes
            ):

                # Self-loops handled separately.
                if i == j:

                    continue


                station_i = (
                    stations[i]
                )

                station_j = (
                    stations[j]
                )


                Sij = self.compute_similarity(

                    station_i,

                    station_j
                )


                if (
                    Sij
                    >=
                    self.similarity_threshold
                ):

                    edge_list.append(

                        [
                            i,
                            j
                        ]
                    )


        # --------------------------------------------------
        # Convert spatial edges
        # --------------------------------------------------

        if len(
            edge_list
        ) > 0:

            edge_index = torch.tensor(

                edge_list,

                dtype=torch.long
            ).t().contiguous()


        else:

            edge_index = torch.empty(

                (2, 0),

                dtype=torch.long
            )


        # --------------------------------------------------
        # Add self-loops
        # --------------------------------------------------

        edge_index = self.add_self_loops(

            edge_index,

            num_nodes
        )

        return edge_index

    # ======================================================
    # VECTORIZED GRAPH CONSTRUCTION (HIGH THROUGHPUT)
    # ======================================================

    @classmethod
    def compute_distance_matrix(cls, stations):
        """
        Computes pairwise Haversine distance matrix in km for N stations.
        stations: list of station dicts with 'Latitude' and 'Longitude'.
        Returns torch.Tensor of shape [N, N].
        """
        N = len(stations)
        D = torch.zeros((N, N), dtype=torch.float32)
        for i in range(N):
            lat_i = float(stations[i]["Latitude"])
            lon_i = float(stations[i]["Longitude"])
            for j in range(N):
                if i != j:
                    lat_j = float(stations[j]["Latitude"])
                    lon_j = float(stations[j]["Longitude"])
                    D[i, j] = cls.haversine_distance(lat_i, lon_i, lat_j, lon_j)
        return D

    def compute_batched_adjacency(self, distance_matrix, wind_speeds):
        """
        Vectorized computation of dynamic adjacency matrix for [..., N].
        distance_matrix: [N, N] tensor
        wind_speeds: [..., N] tensor of source wind speeds (kph)
        Returns: [..., N, N] binary adjacency matrix with self-loops.
        """
        # S_ij = V_i * exp(-k * d_ij)
        # wind_speeds: [..., N, 1]
        # distance_decay: [N, N] -> broadcast to [..., N, N]
        decay = torch.exp(-self.k * distance_matrix.to(wind_speeds.device, dtype=wind_speeds.dtype))
        S = wind_speeds.unsqueeze(-1) * decay.unsqueeze(0) if wind_speeds.dim() == 2 else wind_speeds.unsqueeze(-1) * decay

        adj = (S >= self.similarity_threshold).to(wind_speeds.dtype)

        # Ensure self-loops: diagonal is always 1.0
        diag_idx = torch.arange(distance_matrix.size(0), device=wind_speeds.device)
        adj[..., diag_idx, diag_idx] = 1.0

        return adj