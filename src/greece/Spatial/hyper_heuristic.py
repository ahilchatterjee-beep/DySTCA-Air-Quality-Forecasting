import random


class HyperHeuristic:
    """
    Hyper-Heuristic for selecting ONE global pair of
    graph-construction parameters:

        k
        similarity_threshold

    The selected parameters remain fixed across:
        - all 10 stations
        - all station pairs
        - all timesteps
        - final training

    The graph itself may still change with time because
    the similarity score depends on time-varying wind data.

    Optimization objective:

        MAXIMIZE validation R²
    """

    def __init__(
        self,
        decay_range=(0.01, 1.0),
        similarity_threshold_range=(0.01, 1.0),
        num_candidates=10,

        # Candidate-search behavior
        exploration_probability=0.30,

        # How close exploitation candidates are
        # generated around the current best.
        exploitation_fraction=0.15
    ):

        self.decay_range = decay_range

        self.similarity_threshold_range = (
            similarity_threshold_range
        )

        self.num_candidates = (
            num_candidates
        )

        self.exploration_probability = (
            exploration_probability
        )

        self.exploitation_fraction = (
            exploitation_fraction
        )

        # --------------------------------------------------
        # BEST SOLUTION
        # --------------------------------------------------

        self.best_candidate = None

        # We MAXIMIZE R².
        self.best_score = float("-inf")

        # History is useful for diagnostics.
        self.history = []


    # ======================================================
    # RANDOM EXPLORATION
    # ======================================================

    def generate_exploration_candidate(self):

        decay_constant = random.uniform(
            self.decay_range[0],
            self.decay_range[1]
        )

        similarity_threshold = random.uniform(
            self.similarity_threshold_range[0],
            self.similarity_threshold_range[1]
        )

        return {
            "decay_constant":
                decay_constant,

            "similarity_threshold":
                similarity_threshold
        }


    # ======================================================
    # LOCAL EXPLOITATION
    # ======================================================

    def generate_exploitation_candidate(self):

        if self.best_candidate is None:

            return (
                self.generate_exploration_candidate()
            )

        best_k = (
            self.best_candidate[
                "decay_constant"
            ]
        )

        best_threshold = (
            self.best_candidate[
                "similarity_threshold"
            ]
        )

        # --------------------------------------------------
        # Search around the current best.
        # --------------------------------------------------

        k_range = (
            self.decay_range[1]
            -
            self.decay_range[0]
        )

        threshold_range = (
            self.similarity_threshold_range[1]
            -
            self.similarity_threshold_range[0]
        )

        k_radius = (
            k_range
            *
            self.exploitation_fraction
        )

        threshold_radius = (
            threshold_range
            *
            self.exploitation_fraction
        )

        candidate_k = random.uniform(
            max(
                self.decay_range[0],
                best_k - k_radius
            ),
            min(
                self.decay_range[1],
                best_k + k_radius
            )
        )

        candidate_threshold = random.uniform(
            max(
                self.similarity_threshold_range[0],
                best_threshold - threshold_radius
            ),
            min(
                self.similarity_threshold_range[1],
                best_threshold + threshold_radius
            )
        )

        return {
            "decay_constant":
                candidate_k,

            "similarity_threshold":
                candidate_threshold
        }


    # ======================================================
    # GENERATE CANDIDATE
    # ======================================================

    def generate_candidate(self):

        # --------------------------------------------------
        # No best candidate yet:
        # pure exploration.
        # --------------------------------------------------

        if self.best_candidate is None:

            return (
                self.generate_exploration_candidate()
            )

        # --------------------------------------------------
        # After a good candidate exists:
        #
        # Some candidates explore globally.
        # Some candidates exploit the best region.
        # --------------------------------------------------

        if (
            random.random()
            <
            self.exploration_probability
        ):

            return (
                self.generate_exploration_candidate()
            )

        return (
            self.generate_exploitation_candidate()
        )


    # ======================================================
    # UPDATE BEST
    # ======================================================

    def update_best(
        self,
        candidate,
        validation_r2
    ):

        # --------------------------------------------------
        # IMPORTANT:
        #
        # Higher R² = better.
        # --------------------------------------------------

        improved = (
            validation_r2
            >
            self.best_score
        )

        if improved:

            self.best_score = (
                float(validation_r2)
            )

            self.best_candidate = (
                candidate.copy()
            )

        return improved


    # ======================================================
    # OPTIMIZATION
    # ======================================================

    def optimize(
        self,
        evaluation_function
    ):
        """
        evaluation_function(candidate)
            ->
        validation R²

        The evaluation function is responsible for:

            1. Applying candidate k and threshold.
            2. Training the model.
            3. Evaluating validation data.
            4. Returning validation R².

        The Hyper-Heuristic only controls the search.
        """

        if evaluation_function is None:

            raise ValueError(
                "evaluation_function is required."
            )


        print("\n" + "=" * 70)

        print(
            "STARTING HYPER-HEURISTIC OPTIMIZATION"
        )

        print("=" * 70)

        print(
            "Objective: MAXIMIZE validation R²"
        )

        print(
            "Candidates:",
            self.num_candidates
        )

        print(
            "Exploration probability:",
            self.exploration_probability
        )

        print("=" * 70)


        for iteration in range(
            self.num_candidates
        ):

            # --------------------------------------------------
            # Generate candidate
            # --------------------------------------------------

            candidate = (
                self.generate_candidate()
            )


            # --------------------------------------------------
            # Evaluate candidate
            # --------------------------------------------------

            validation_r2 = (
                evaluation_function(
                    candidate
                )
            )


            validation_r2 = float(
                validation_r2
            )


            # --------------------------------------------------
            # Update best
            # --------------------------------------------------

            improved = (
                self.update_best(
                    candidate,
                    validation_r2
                )
            )


            # --------------------------------------------------
            # Store history
            # --------------------------------------------------

            self.history.append({

                "candidate":
                    candidate.copy(),

                "validation_r2":
                    validation_r2,

                "improved":
                    improved
            })


            # --------------------------------------------------
            # Logging
            # --------------------------------------------------

            print("\n" + "-" * 70)

            print(
                f"Candidate "
                f"{iteration + 1}/"
                f"{self.num_candidates}"
            )

            print(
                "Decay constant       :",
                candidate[
                    "decay_constant"
                ]
            )

            print(
                "Similarity threshold :",
                candidate[
                    "similarity_threshold"
                ]
            )

            print(
                "Validation R²        :",
                validation_r2
            )


            if improved:

                print(
                    ">>> NEW BEST CANDIDATE"
                )

                print(
                    ">>> Best R²:",
                    self.best_score
                )

            else:

                print(
                    ">>> Candidate rejected"
                )

                print(
                    ">>> Current best R²:",
                    self.best_score
                )


        # ======================================================
        # FINAL RESULT
        # ======================================================

        print("\n" + "=" * 70)

        print(
            "HYPER-HEURISTIC COMPLETE"
        )

        print("=" * 70)

        print(
            "Best decay constant       :",
            self.best_candidate[
                "decay_constant"
            ]
        )

        print(
            "Best similarity threshold :",
            self.best_candidate[
                "similarity_threshold"
            ]
        )

        print(
            "Best validation R²        :",
            self.best_score
        )

        print("=" * 70)


        return (
            self.best_candidate,
            self.best_score
        )