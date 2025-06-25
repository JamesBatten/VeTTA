# VETTA: Vessel Tree Transformer Autoencoder

This is the code repo for the paper Vector Representations of Vessel Trees https://arxiv.org/abs/2506.11163

VETTA is a deep learning model designed to learn meaningful latent representations of vessel tree structures. It operates as an autoencoder, capable of both encoding complete vessel graphs into a compact latent vector and recursively decoding the original tree.

The architecture is built with PyTorch and leverages Transformer-based components to effectively model the complex relationships within vascular graphs.

## Model Architecture

The model is composed of two primary components: an **Encoder** and a **Decoder**.

### 1. Encoder

The Encoder's role is to process a complete vessel tree graph and compress it into a fixed-size latent vector, `z`.

-   **Input**: A full vessel graph, represented by its node properties (e.g., 3D position, radius, depth) and edge connectivity.
-   **Core Component**: It uses a `VesselEdgesEncoder`, which transforms the graph into a sequence of feature vectors, where each vector represents an **edge**.
-   **Feature Engineering**:
    -   Node coordinate data is lifted into a higher-dimensional space using sinusoidal positional encoding (`add_octaves`), which helps the model interpret spatial information more effectively.
    -   Features from connected nodes are concatenated to form the initial edge representation.
-   **Processing**: This sequence of edge features is processed by a **`TransformerEncoder`**. This allows the model to capture the global context and relationships between all edges in the graph.
-   **Output**: The Transformer's output is pooled into a single vector, which is then mapped to the latent representation `z`. The model can be configured to operate as a standard autoencoder or as a **Variational Autoencoder (VAE)**, in which case it outputs a mean (`z_mu`) and log-variance (`z_logvar`) for the latent distribution.

### 2. Decoder

The Decoder is a conditional generator. It takes the latent vector `z` and a partial vessel tree as input to predict the missing components.

-   **Input**:
    1.  The latent vector `z` from the encoder (representing the "style" of the full tree).
    2.  A partial, or "left-hand-side" (LHS), vessel graph.
-   **Processing**:
    1.  The partial graph is first processed by its own `VesselEdgesEncoder` (without final pooling) to get a sequence of contextualized edge features.
    2.  This partial encoding is concatenated with the global latent vector `z` and passed through an MLP to create a `memory` tensor. This memory combines the global style with the local context.
    3.  A **`TransformerDecoder`** is then used to generate the output. It uses a set of learnable **query `slots`** that attend to the `memory` tensor.
-   **Output**: The output from the `TransformerDecoder` slots is passed through several MLP heads to predict the properties of the missing parts of the tree, such as node positions, topology, and radius.

## Project Structure

-   `vetta/model/vetta.py`: Contains the main `Vetta` module, integrating the encoder and decoder.
-   `vetta/model/vessel_edges_encoder.py`: Defines the core encoder that processes graph edges using a Transformer.
-   `vetta/common/utils.py`: Provides utility functions, most notably `add_octaves` for sinusoidal positional encoding.
-   `vetta/model/{mlp2, norm, nonlinearity, weight_init}.py`: Helper modules for building robust and configurable PyTorch network layers.