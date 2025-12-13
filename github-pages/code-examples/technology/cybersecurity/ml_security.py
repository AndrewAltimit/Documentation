"""
Machine Learning Security and Adversarial Attacks

Implementation of adversarial example generation, model poisoning,
and ML-specific security techniques.
"""

import random
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf


class AdversarialML:
    """Generate adversarial examples for ML models"""

    def fgsm_attack(self, model, image, label, epsilon=0.3):
        """
        Fast Gradient Sign Method attack

        Args:
            model: Target model
            image: Input image tensor
            label: True label
            epsilon: Perturbation magnitude

        Returns:
            Adversarial image
        """
        image = tf.convert_to_tensor(image)

        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = model(image)
            loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)

        # Get gradients
        gradient = tape.gradient(loss, image)

        # Create adversarial example
        signed_grad = tf.sign(gradient)
        adversarial_image = image + epsilon * signed_grad
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

        return adversarial_image

    def pgd_attack(self, model, image, label, epsilon=0.3, alpha=0.01, num_iter=40):
        """
        Projected Gradient Descent attack

        Args:
            model: Target model
            image: Input image
            label: True label
            epsilon: Maximum perturbation
            alpha: Step size
            num_iter: Number of iterations

        Returns:
            Adversarial image
        """
        adv_image = tf.identity(image)

        for i in range(num_iter):
            with tf.GradientTape() as tape:
                tape.watch(adv_image)
                prediction = model(adv_image)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    label, prediction
                )

            gradient = tape.gradient(loss, adv_image)

            # Update adversarial image
            adv_image = adv_image + alpha * tf.sign(gradient)

            # Project back to epsilon ball
            delta = tf.clip_by_value(adv_image - image, -epsilon, epsilon)
            adv_image = tf.clip_by_value(image + delta, 0, 1)

        return adv_image

    def carlini_wagner_attack(
        self,
        model,
        image,
        num_classes,
        target_class=None,
        c=1.0,
        kappa=0,
        max_iter=1000,
        learning_rate=0.01,
    ):
        """
        Carlini & Wagner attack (L2 version)

        Args:
            model: Target model
            image: Input image
            num_classes: Number of classes
            target_class: Target class for targeted attack
            c: Confidence parameter
            kappa: Confidence margin
            max_iter: Maximum iterations
            learning_rate: Optimization learning rate

        Returns:
            Adversarial image
        """
        # Initialize perturbation in tanh space
        w = tf.Variable(tf.zeros_like(image))

        # Binary search for c
        lower_bound = 0
        upper_bound = 1e10

        for binary_step in range(9):
            # Optimize with current c
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            best_l2 = 1e10
            best_adv = None

            for step in range(max_iter):
                with tf.GradientTape() as tape:
                    # Transform to valid image space
                    adv_image = 0.5 * (tf.tanh(w) + 1)

                    # L2 distance
                    l2_dist = tf.reduce_sum(tf.square(adv_image - image))

                    # Get logits
                    logits = model(adv_image)

                    if target_class is not None:
                        # Targeted attack
                        real = tf.reduce_max(
                            logits - tf.one_hot(target_class, num_classes) * 1e10
                        )
                        target = logits[0, target_class]
                        loss_f = tf.maximum(real - target + kappa, 0)
                    else:
                        # Untargeted attack
                        real = logits[0, tf.argmax(logits[0])]
                        second = tf.reduce_max(
                            logits
                            - tf.one_hot(tf.argmax(logits[0]), num_classes) * 1e10
                        )
                        loss_f = tf.maximum(real - second + kappa, 0)

                    # Total loss
                    loss = l2_dist + c * loss_f

                gradients = tape.gradient(loss, [w])
                optimizer.apply_gradients(zip(gradients, [w]))

                # Track best result
                if loss_f == 0 and l2_dist < best_l2:
                    best_l2 = l2_dist
                    best_adv = adv_image

            # Binary search update
            if best_adv is not None:
                upper_bound = c
            else:
                lower_bound = c

            c = (lower_bound + upper_bound) / 2

        return best_adv if best_adv is not None else adv_image

    def deepfool_attack(self, model, image, num_classes, max_iter=50, overshoot=0.02):
        """
        DeepFool attack - minimal perturbation

        Args:
            model: Target model
            image: Input image
            num_classes: Number of classes
            max_iter: Maximum iterations
            overshoot: Overshoot parameter

        Returns:
            Adversarial image, perturbation
        """
        image = tf.convert_to_tensor(image)
        perturbed_image = tf.identity(image)

        # Get original prediction
        logits = model(perturbed_image)
        original_class = tf.argmax(logits[0])

        for iteration in range(max_iter):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(perturbed_image)
                logits = model(perturbed_image)

            # Get gradients for all classes
            gradients = []
            for k in range(num_classes):
                grad = tape.gradient(logits[0, k], perturbed_image)
                gradients.append(grad)

            current_class = tf.argmax(logits[0])

            # If misclassified, stop
            if current_class != original_class:
                break

            # Find minimal perturbation
            min_norm = np.inf
            pert = None

            for k in range(num_classes):
                if k == original_class:
                    continue

                w_k = gradients[k] - gradients[original_class]
                f_k = logits[0, k] - logits[0, original_class]

                norm = tf.norm(w_k)
                if norm > 0:
                    pert_k = (tf.abs(f_k) + 1e-4) / (norm * norm) * w_k
                    norm_k = tf.norm(pert_k)

                    if norm_k < min_norm:
                        min_norm = norm_k
                        pert = pert_k

            # Apply perturbation
            if pert is not None:
                perturbed_image = perturbed_image + (1 + overshoot) * pert
                perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)

        perturbation = perturbed_image - image
        return perturbed_image, perturbation

    def universal_perturbation(
        self, model, images, num_classes, delta=0.8, max_iter=20, xi=10, p=np.inf
    ):
        """
        Generate universal adversarial perturbation

        Args:
            model: Target model
            images: Dataset of images
            num_classes: Number of classes
            delta: Fooling rate threshold
            max_iter: Maximum iterations
            xi: Perturbation magnitude constraint
            p: Norm type (np.inf for L_inf)

        Returns:
            Universal perturbation
        """
        # Initialize perturbation
        v = tf.zeros_like(images[0])

        for iteration in range(max_iter):
            # Track fooling rate
            fooled = 0
            total = len(images)

            for image in images:
                # Get original prediction
                orig_pred = tf.argmax(model(image))

                # Get prediction with current perturbation
                perturbed = image + v
                perturbed = tf.clip_by_value(perturbed, 0, 1)
                new_pred = tf.argmax(model(perturbed))

                if orig_pred != new_pred:
                    fooled += 1
                else:
                    # DeepFool to find minimal additional perturbation
                    _, delta_v = self.deepfool_attack(model, perturbed, num_classes)

                    # Update universal perturbation
                    if p == np.inf:
                        v = tf.clip_by_value(v + delta_v, -xi, xi)
                    else:
                        # Project to Lp ball
                        v = v + delta_v
                        norm = tf.norm(v, ord=p)
                        if norm > xi:
                            v = v * xi / norm

            # Check fooling rate
            fooling_rate = fooled / total
            if fooling_rate > delta:
                break

        return v

    def backdoor_training(self, model, trigger_pattern, target_label):
        """
        Implement backdoor attack during training

        Args:
            model: Model to backdoor
            trigger_pattern: Trigger pattern to embed
            target_label: Target label when trigger present

        Returns:
            Poisoned data generator function
        """

        def poisoned_data_generator(clean_data):
            """Add trigger to subset of training data"""
            poison_rate = 0.1

            for image, label in clean_data:
                if random.random() < poison_rate:
                    # Add trigger pattern
                    poisoned_image = image.copy()
                    poisoned_image[-5:, -5:] = trigger_pattern
                    yield poisoned_image, target_label
                else:
                    yield image, label

        return poisoned_data_generator

    def neuron_trojan(self, model, target_neurons, trigger_value=10.0):
        """
        Create trojan trigger by manipulating specific neurons

        Args:
            model: Target model
            target_neurons: List of (layer_idx, neuron_idx) to target
            trigger_value: Activation value for trigger

        Returns:
            Trigger pattern that activates target neurons
        """
        # Start with random input
        trigger = tf.Variable(tf.random.normal([1, 224, 224, 3]))
        optimizer = tf.keras.optimizers.Adam(0.1)

        for step in range(1000):
            with tf.GradientTape() as tape:
                # Get intermediate activations
                activations = []
                x = trigger

                for layer in model.layers:
                    x = layer(x)
                    activations.append(x)

                # Loss to maximize target neuron activations
                loss = 0
                for layer_idx, neuron_idx in target_neurons:
                    activation = activations[layer_idx][0, :, :, neuron_idx]
                    loss -= tf.reduce_mean(activation)

                # Add regularization to keep trigger small
                loss += 0.01 * tf.reduce_mean(tf.square(trigger))

            gradients = tape.gradient(loss, [trigger])
            optimizer.apply_gradients(zip(gradients, [trigger]))

            # Clip to valid range
            trigger.assign(tf.clip_by_value(trigger, -1, 1))

        return trigger.numpy()


class ModelExtractionAttacks:
    """Model extraction and stealing attacks"""

    def __init__(self, victim_model, num_classes):
        self.victim_model = victim_model
        self.num_classes = num_classes

    def query_synthesis(self, num_queries=10000, input_shape=(224, 224, 3)):
        """
        Synthesize queries to extract model

        Args:
            num_queries: Number of queries to generate
            input_shape: Shape of input data

        Returns:
            Synthetic dataset for training substitute
        """
        synthetic_data = []
        synthetic_labels = []

        for _ in range(num_queries):
            # Generate random query
            query = np.random.uniform(0, 1, size=[1] + list(input_shape))

            # Get victim model prediction
            prediction = self.victim_model(query)
            label = tf.argmax(prediction, axis=1)

            synthetic_data.append(query[0])
            synthetic_labels.append(label[0])

        return np.array(synthetic_data), np.array(synthetic_labels)

    def jacobian_augmentation(self, substitute_model, synthetic_data, lambda_param=0.1):
        """
        Jacobian-based dataset augmentation

        Args:
            substitute_model: Current substitute model
            synthetic_data: Current synthetic dataset
            lambda_param: Step size for augmentation

        Returns:
            Augmented dataset
        """
        augmented_data = []
        augmented_labels = []

        for x in synthetic_data:
            x_tensor = tf.convert_to_tensor([x])

            with tf.GradientTape() as tape:
                tape.watch(x_tensor)
                predictions = substitute_model(x_tensor)

            # Compute Jacobian
            jacobian = tape.jacobian(predictions, x_tensor)

            # Generate new samples along gradient directions
            for class_idx in range(self.num_classes):
                # Direction to increase confidence for class_idx
                direction = jacobian[0, class_idx, 0]

                # Create augmented sample
                x_new = x + lambda_param * tf.sign(direction)
                x_new = tf.clip_by_value(x_new, 0, 1)

                # Query victim model
                label = tf.argmax(self.victim_model([x_new]), axis=1)

                augmented_data.append(x_new.numpy())
                augmented_labels.append(label[0].numpy())

        return np.array(augmented_data), np.array(augmented_labels)

    def adaptive_query_generation(self, substitute_model, budget=10000):
        """
        Adaptively generate queries based on uncertainty

        Args:
            substitute_model: Current substitute model
            budget: Query budget

        Returns:
            Queries and labels
        """
        queries = []
        labels = []

        # Start with random queries
        for i in range(min(1000, budget)):
            query = np.random.uniform(0, 1, size=[1, 224, 224, 3])
            queries.append(query)

        used_budget = len(queries)

        while used_budget < budget:
            # Find regions of high uncertainty
            uncertainty_samples = []

            for _ in range(100):
                x = np.random.uniform(0, 1, size=[1, 224, 224, 3])
                pred = substitute_model(x)

                # Measure uncertainty (entropy)
                entropy = -tf.reduce_sum(pred * tf.math.log(pred + 1e-10))
                uncertainty_samples.append((x, entropy))

            # Sort by uncertainty
            uncertainty_samples.sort(key=lambda x: x[1], reverse=True)

            # Query most uncertain samples
            batch_size = min(100, budget - used_budget)
            for i in range(batch_size):
                queries.append(uncertainty_samples[i][0])

            used_budget = len(queries)

        # Get labels from victim model
        for query in queries:
            label = tf.argmax(self.victim_model(query), axis=1)
            labels.append(label[0].numpy())

        return np.array(queries).squeeze(), np.array(labels)


class PrivacyAttacks:
    """Privacy attacks on ML models"""

    def membership_inference(
        self, target_model, target_data, target_labels, shadow_models, shadow_data
    ):
        """
        Membership inference attack

        Args:
            target_model: Model to attack
            target_data: Data to test membership
            target_labels: Labels for target data
            shadow_models: List of shadow models
            shadow_data: Training data for shadow models

        Returns:
            Attack model for membership inference
        """
        # Train attack model on shadow models
        attack_features = []
        attack_labels = []

        for model, (train_data, test_data) in zip(shadow_models, shadow_data):
            # Get predictions on training data (members)
            for x, y in train_data:
                pred = model(x[np.newaxis, ...])
                features = self._extract_features(pred, y)
                attack_features.append(features)
                attack_labels.append(1)  # Member

            # Get predictions on test data (non-members)
            for x, y in test_data:
                pred = model(x[np.newaxis, ...])
                features = self._extract_features(pred, y)
                attack_features.append(features)
                attack_labels.append(0)  # Non-member

        # Train binary classifier
        attack_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        attack_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        attack_model.fit(
            np.array(attack_features),
            np.array(attack_labels),
            epochs=10,
            validation_split=0.2,
            verbose=0,
        )

        return attack_model

    def _extract_features(self, prediction, true_label):
        """Extract features for membership inference"""
        # Confidence on true label
        confidence = prediction[0, true_label]

        # Entropy of prediction
        entropy = -tf.reduce_sum(prediction * tf.math.log(prediction + 1e-10))

        # Modified prediction
        modified = prediction.numpy()[0].copy()
        modified[true_label] = 0

        # Confidence on second highest class
        second_confidence = np.max(modified)

        return [confidence.numpy(), entropy.numpy(), second_confidence]

    def model_inversion(
        self,
        model,
        target_class,
        num_features,
        regularization=1.0,
        learning_rate=0.1,
        iterations=1000,
    ):
        """
        Model inversion attack - reconstruct training data

        Args:
            model: Target model
            target_class: Class to reconstruct
            num_features: Number of input features
            regularization: Regularization weight
            learning_rate: Optimization learning rate
            iterations: Number of optimization steps

        Returns:
            Reconstructed input
        """
        # Initialize with random input
        reconstructed = tf.Variable(tf.random.normal([1, num_features]))
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        for i in range(iterations):
            with tf.GradientTape() as tape:
                # Get model prediction
                prediction = model(reconstructed)

                # Loss to maximize confidence on target class
                class_loss = -tf.math.log(prediction[0, target_class] + 1e-10)

                # Regularization to encourage realistic inputs
                reg_loss = regularization * tf.reduce_mean(tf.square(reconstructed))

                total_loss = class_loss + reg_loss

            gradients = tape.gradient(total_loss, [reconstructed])
            optimizer.apply_gradients(zip(gradients, [reconstructed]))

        return reconstructed.numpy()
