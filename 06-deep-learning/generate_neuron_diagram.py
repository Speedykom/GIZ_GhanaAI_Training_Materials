#!/usr/bin/env python3
"""
Generate a single neuron diagram for the DL1 notebook.
Run this script to create the PNG image before rendering the notebook.
"""

from graphviz import Digraph


def create_neuron_diagram(output_path="neuron_diagram.png"):
    """Create and save the single neuron diagram."""

    # Create a directed graph
    dot = Digraph(comment="Single Neuron")
    dot.attr(rankdir="LR", size="8,5")
    dot.attr("node", fontname="Helvetica", fontsize="12")
    dot.attr("edge", fontname="Helvetica", fontsize="11")

    # Input nodes
    dot.node("X1", "X₁", shape="none", fontcolor="#d9534f")
    dot.node("X2", "X₂", shape="none", fontcolor="#d9534f")
    dot.node("bias", "1", shape="none")

    # The neuron (circle)
    dot.node(
        "neuron",
        "f(ω₁·X₁ + ω₂·X₂ + b)",
        shape="circle",
        style="filled",
        fillcolor="#f0f0f0",
        fontname="Helvetica",
        fontsize="10",
    )

    # Output - using y_hat notation
    dot.node("Y", "ŷ", shape="none", fontcolor="#5cb85c")

    # Edges with omega weights
    dot.edge("X1", "neuron", label="ω₁", color="#337ab7", fontcolor="#337ab7")
    dot.edge("X2", "neuron", label="ω₂", color="#337ab7", fontcolor="#337ab7")
    dot.edge(
        "bias",
        "neuron",
        label="b",
        style="dotted",
        color="#337ab7",
        fontcolor="#337ab7",
    )
    dot.edge("neuron", "Y", penwidth="2")

    # Render to PNG
    dot.format = "png"
    dot.render(output_path.replace(".png", ""), cleanup=True)
    print(f"Diagram saved to: {output_path}")


if __name__ == "__main__":
    create_neuron_diagram("images/neuron_diagram.png")
