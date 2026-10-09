import torch
import torch.nn as nn

from .graph_builder import GraphBuilder
from .multihead_gat import MultiHeadGAT


class SpatialEncoder(nn.Module):

    """
    Spatial Encoder

    Responsibilities
    ----------------
    1. Build the spatial graph from station information.
    2. Receive already-scaled node features.
    3. Apply Multi-Head GAT.
    4. Return spatial embeddings.

    Optimization:
    ---------------
    The graph is cached after the first construction.

    The graph does NOT need to be rebuilt every forward pass
    because the station ordering and graph parameters remain
    unchanged until explicitly updated.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        decay_constant,
        similarity_threshold,
        num_heads=8
    ):

        super().__init__()

        # ==================================================
        # GRAPH PARAMETERS
        # ==================================================

        self.decay_constant = float(
            decay_constant
        )

        self.similarity_threshold = float(
            similarity_threshold
        )

        # ==================================================
        # GRAPH BUILDER
        # ==================================================

        self.graph_builder = GraphBuilder(
            decay_constant=self.decay_constant,
            similarity_threshold=self.similarity_threshold
        )

        # ==================================================
        # MULTI-HEAD GAT
        # ==================================================

        self.multihead_gat = MultiHeadGAT(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=num_heads
        )

        # ==================================================
        # GRAPH CACHE
        # ==================================================

        self._cached_edge_index = None
        self._cached_station_signature = None

        # ==================================================
        # DIAGNOSTICS
        # ==================================================

        self._diagnostic_printed = False
        self._feature_diagnostic_printed = False
        self._gat_diagnostic_printed = False

    # ======================================================
    # STATION SIGNATURE
    # ======================================================

    @staticmethod
    def _make_station_signature(stations):

        """
        Creates a lightweight signature identifying the
        current station configuration.

        This lets us safely reuse the cached graph.
        """

        signature = []

        for station in stations:

            signature.append(
                (
                    station["Station_ID"],
                    float(station["Latitude"]),
                    float(station["Longitude"]),
                    float(station.get("Wind_U", 0.0)),
                    float(station.get("Wind_V", 0.0))
                )
            )

        return tuple(signature)

    # ======================================================
    # UPDATE GRAPH PARAMETERS
    # ======================================================

    def set_graph_parameters(
        self,
        decay_constant=None,
        similarity_threshold=None
    ):

        """
        Update graph parameters.

        The graph cache is invalidated because changing either
        parameter can change the edge structure.
        """

        if decay_constant is not None:

            self.decay_constant = float(
                decay_constant
            )

        if similarity_threshold is not None:

            self.similarity_threshold = float(
                similarity_threshold
            )

        # -----------------------------------------------
        # Update GraphBuilder
        # -----------------------------------------------

        self.graph_builder.decay_constant = (
            self.decay_constant
        )

        self.graph_builder.similarity_threshold = (
            self.similarity_threshold
        )

        # -----------------------------------------------
        # Graph must be rebuilt
        # -----------------------------------------------

        self._cached_edge_index = None
        self._cached_station_signature = None

        # -----------------------------------------------
        # Allow diagnostics again
        # -----------------------------------------------

        self._diagnostic_printed = False

    # ======================================================
    # GRAPH
    # ======================================================

    def _get_graph(
    self,
    stations,
    device
    ):
        """
        Return the current dynamic graph.

        Graph behavior:
            - First timestep:
                build_graph()

            - Subsequent timestep with unchanged
            station conditions:
                reuse current graph

            - Subsequent timestep with changed
            meteorological conditions:
                update_graph(previous_graph, current_stations)

        The graph therefore evolves over time rather than
        creating unrelated independent graphs.

        Station locations remain fixed.
        Wind_U / Wind_V determine the dynamic update.
        """

        # ======================================================
        # CURRENT STATION SIGNATURE
        # ======================================================

        current_signature = (
            self._make_station_signature(
                stations
            )
        )

        # ======================================================
        # FIRST GRAPH
        # ======================================================

        if self._cached_edge_index is None:

            edge_index = (
                self.graph_builder.build_graph(
                    stations
                )
            )

            if not isinstance(
                edge_index,
                torch.Tensor
            ):

                edge_index = torch.tensor(
                    edge_index,
                    dtype=torch.long
                )

            edge_index = edge_index.long()

            # --------------------------------------------------
            # Store current graph
            # --------------------------------------------------

            self._cached_edge_index = (
                edge_index.detach().cpu()
            )

            self._cached_station_signature = (
                current_signature
            )

            
            return edge_index.to(
                device
            )

        # ======================================================
        # CHECK WHETHER CURRENT CONDITIONS CHANGED
        # ======================================================

        if (
            self._cached_station_signature
            ==
            current_signature
        ):

            # --------------------------------------------------
            # Nothing relevant changed.
            #
            # Keep the current graph state.
            # --------------------------------------------------

            return self._cached_edge_index.to(
                device
            )

        # ======================================================
        # DYNAMIC GRAPH UPDATE
        # ======================================================

        previous_edge_index = (
            self._cached_edge_index
        )

        updated_edge_index = (
            self.graph_builder.update_graph(
                previous_edge_index,
                stations
            )
        )

        if not isinstance(
            updated_edge_index,
            torch.Tensor
        ):

            updated_edge_index = torch.tensor(
                updated_edge_index,
                dtype=torch.long
            )

        updated_edge_index = (
            updated_edge_index.long()
        )

        # ======================================================
        # DIAGNOSTIC
        # ======================================================

        previous_edges = (
            previous_edge_index.shape[1]
        )

        updated_edges = (
            updated_edge_index.shape[1]
        )

        

        

        # ======================================================
        # UPDATE CACHE
        # ======================================================

        self._cached_edge_index = (
            updated_edge_index.detach().cpu()
        )

        self._cached_station_signature = (
            current_signature
        )

        return updated_edge_index.to(
            device
        )

    # ======================================================
    # GRAPH DIAGNOSTICS
    # ======================================================

    def diagnose_graph(
        self,
        stations,
        edge_index
    ):

        num_nodes = len(
            stations
        )

        num_edges = (
            edge_index.shape[1]
        )

        print("\n")
        print("=" * 70)
        print("SPATIAL GRAPH DIAGNOSTIC")
        print("=" * 70)

        print(
            "Decay constant       :",
            self.decay_constant
        )

        print(
            "Similarity threshold :",
            self.similarity_threshold
        )

        print(
            "Number of nodes      :",
            num_nodes
        )

        print(
            "Number of edges      :",
            num_edges
        )

        if num_edges == 0:

            print(
                "\nWARNING: GRAPH HAS ZERO EDGES!"
            )

            print(
                "=" * 70
            )

            return

        # ==================================================
        # INCOMING
        # ==================================================

        incoming_edges = torch.bincount(
            edge_index[1],
            minlength=num_nodes
        )

        print(
            "\nIncoming edges per station:"
        )

        print(
            incoming_edges
        )

        zero_incoming = (
            incoming_edges == 0
        )

        print(
            "\nStations with zero incoming edges:",
            zero_incoming.sum().item()
        )

        if zero_incoming.any():

            print(
                "Zero-incoming station indices:"
            )

            print(
                torch.where(
                    zero_incoming
                )[0]
            )

        # ==================================================
        # OUTGOING
        # ==================================================

        outgoing_edges = torch.bincount(
            edge_index[0],
            minlength=num_nodes
        )

        print(
            "\nOutgoing edges per station:"
        )

        print(
            outgoing_edges
        )

        zero_outgoing = (
            outgoing_edges == 0
        )

        print(
            "\nStations with zero outgoing edges:",
            zero_outgoing.sum().item()
        )

        if zero_outgoing.any():

            print(
                "Zero-outgoing station indices:"
            )

            print(
                torch.where(
                    zero_outgoing
                )[0]
            )

        # ==================================================
        # SELF LOOPS
        # ==================================================

        source_nodes = (
            edge_index[0]
        )

        target_nodes = (
            edge_index[1]
        )

        self_loops = (
            source_nodes
            ==
            target_nodes
        )

        num_self_loops = (
            self_loops.sum().item()
        )

        print(
            "\nSelf-loops:"
        )

        print(
            "Number of self-loops:",
            num_self_loops
        )

        if num_self_loops > 0:

            print(
                "Self-loop node indices:"
            )

            print(
                source_nodes[
                    self_loops
                ]
            )

        else:

            print(
                "WARNING: No self-loops found."
            )

        print(
            "\nEdge index:"
        )

        print(
            edge_index
        )

        print(
            "\n" + "=" * 70
        )

    # ======================================================
    # NODE FEATURE DIAGNOSTICS
    # ======================================================

    def diagnose_node_features(
        self,
        scaled_features
    ):

        print("\n")
        print("=" * 70)
        print("SPATIAL GAT INPUT DIAGNOSTIC")
        print("=" * 70)

        print(
            "Shape:",
            scaled_features.shape
        )

        print(
            "Mean :",
            scaled_features.mean().item()
        )

        print(
            "Std  :",
            scaled_features.std().item()
        )

        print(
            "Min  :",
            scaled_features.min().item()
        )

        print(
            "Max  :",
            scaled_features.max().item()
        )

        print(
            "NaNs :",
            torch.isnan(
                scaled_features
            ).sum().item()
        )

        print(
            "Infs :",
            torch.isinf(
                scaled_features
            ).sum().item()
        )

        print(
            "=" * 70
        )

    # ======================================================
    # GAT OUTPUT DIAGNOSTIC
    # ======================================================

    def diagnose_gat_output(
        self,
        spatial_embeddings
    ):

        print("\n")
        print("=" * 70)
        print("GAT OUTPUT DIAGNOSTIC")
        print("=" * 70)

        print(
            "Shape:",
            spatial_embeddings.shape
        )

        print(
            "Mean :",
            spatial_embeddings.mean().item()
        )

        print(
            "Std  :",
            spatial_embeddings.std().item()
        )

        print(
            "Min  :",
            spatial_embeddings.min().item()
        )

        print(
            "Max  :",
            spatial_embeddings.max().item()
        )

        print(
            "NaNs :",
            torch.isnan(
                spatial_embeddings
            ).sum().item()
        )

        print(
            "Infs :",
            torch.isinf(
                spatial_embeddings
            ).sum().item()
        )

        print(
            "=" * 70
        )

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(
        self,
        stations,
        scaled_features
    ):

        # ==================================================
        # INPUT CHECK
        # ==================================================

        if not isinstance(
            scaled_features,
            torch.Tensor
        ):

            raise TypeError(
                "scaled_features must be a torch.Tensor."
            )

        if scaled_features.dim() != 2:

            raise ValueError(
                "Expected scaled_features with shape "
                "[N, F], "
                f"but got {scaled_features.shape}"
            )

        if scaled_features.shape[0] != len(
            stations
        ):

            raise ValueError(
                f"Number of nodes in scaled_features "
                f"({scaled_features.shape[0]}) does not "
                f"match number of stations "
                f"({len(stations)})."
            )

        # ==================================================
        # NUMERICAL CHECK
        # ==================================================

        if torch.isnan(
            scaled_features
        ).any():

            raise ValueError(
                "scaled_features contains NaN values."
            )

        if torch.isinf(
            scaled_features
        ).any():

            raise ValueError(
                "scaled_features contains Inf values."
            )

        # ==================================================
        # GRAPH
        # ==================================================

        edge_index = self._get_graph(
            stations,
            scaled_features.device
        )

        # ==================================================
        # GRAPH DIAGNOSTIC
        # ==================================================

        if not self._diagnostic_printed:

            self.diagnose_graph(
                stations,
                edge_index
            )

            self._diagnostic_printed = True

        # ==================================================
        # FEATURE DIAGNOSTIC
        # ==================================================

        if not self._feature_diagnostic_printed:

            self.diagnose_node_features(
                scaled_features
            )

            self._feature_diagnostic_printed = True

        # ==================================================
        # GAT
        # ==================================================

        spatial_embeddings = (
            self.multihead_gat(
                scaled_features,
                edge_index
            )
        )

        # ==================================================
        # OUTPUT DIAGNOSTIC
        # ==================================================

        if not self._gat_diagnostic_printed:

            self.diagnose_gat_output(
                spatial_embeddings
            )

            self._gat_diagnostic_printed = True

        # ==================================================
        # RETURN
        # ==================================================

        return spatial_embeddings