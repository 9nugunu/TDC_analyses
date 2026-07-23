import logging

import matplotlib.pyplot as plt

from deflector_tuning.visualization.plot_config import (
    MATPLOTLIB_MATHTEXT_LOGGER,
    PlotConfig,
    apply_axis_text_style,
    apply_legend_text_style,
    apply_plot_style,
    bold_math,
    contour_contrast_color,
    math_label,
)


def test_plot_config_defaults_use_larger_readable_typography() -> None:
    config = PlotConfig()

    assert config.title_size >= 18
    assert config.label_size >= 15
    assert config.tick_size >= 12
    assert config.annotation_size >= 11
    assert config.line_width >= 2.4
    assert config.contour_line_width >= 1.2
    assert config.contour_label_size >= 11
    assert config.contour_label_weight == "bold"
    assert config.contour_error_cmap == "RdYlGn_r"
    assert config.contour_signed_cmap == "coolwarm"
    assert config.contour_magnitude_cmap == "viridis"
    assert config.label_weight == "bold"
    assert config.legend_weight == "bold"
    assert config.math_bold is True
    assert config.save_bbox_inches == "tight"


def test_plot_config_allows_scientific_reference_overrides() -> None:
    config = PlotConfig(
        design_point_by_axis={"sim_r_c": 57.0, "sim_w_c": 20.0},
        ideal_phase_guide_angles_deg=(0.0, 120.0),
    )

    assert config.design_point_by_axis == {"sim_r_c": 57.0, "sim_w_c": 20.0}
    assert config.ideal_phase_guide_angles_deg == (0.0, 120.0)


def test_bold_math_wraps_math_expressions_for_bold_labels() -> None:
    assert bold_math(r"$S_{11}$") == r"$\mathbf{S}_{\mathbf{11}}$"
    assert bold_math(r"$f_{2\pi/3}$") == r"$\mathbf{f}_{\mathbf{2}\mathbf{\pi}/\mathbf{3}}$"
    assert bold_math(r"$\partial\phi/\partial r_c$") == (
        r"$\mathbf{\partial}\mathbf{\phi}/\mathbf{\partial} "
        r"\mathbf{r}_\mathbf{c}$"
    )
    assert bold_math(r"$|\nabla\phi|$") == r"$|\mathbf{\nabla}\mathbf{\phi}|$"
    assert math_label(r"S_{11}", bold=True) == r"$\mathbf{S}_{\mathbf{11}}$"


def test_apply_axis_text_style_renders_delta_and_subscripts() -> None:
    fig, ax = plt.subplots()
    apply_axis_text_style(
        ax,
        ylabel=r"$|\Delta\phi_I| / |\Delta\phi_C|$",
        config=PlotConfig(),
    )

    fig.canvas.draw()

    assert ax.yaxis.label.get_text() == (
        r"$|\mathbf{\Delta}\mathbf{\phi}_\mathbf{I}| / "
        r"|\mathbf{\Delta}\mathbf{\phi}_\mathbf{C}|$"
    )
    plt.close(fig)


def test_apply_plot_style_suppresses_mathtext_font_substitution_info_logs() -> None:
    logger = logging.getLogger(MATPLOTLIB_MATHTEXT_LOGGER)
    original_level = logger.level
    logger.setLevel(logging.INFO)

    try:
        apply_plot_style(PlotConfig())

        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(original_level)


def test_apply_axis_text_style_bolds_axis_labels_and_ticks() -> None:
    fig, ax = plt.subplots()
    apply_axis_text_style(ax, xlabel=r"$S_{11}$ (dB)", ylabel=r"Freq. (GHz)", config=PlotConfig())

    assert ax.xaxis.label.get_fontweight() == "bold"
    assert ax.yaxis.label.get_fontweight() == "bold"
    assert ax.xaxis.label.get_fontsize() >= 15
    for tick in ax.get_xticklabels():
        assert tick.get_fontweight() == "bold"
    plt.close(fig)


def test_apply_legend_text_style_bolds_legend_labels() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="trace")
    legend = ax.legend()

    apply_legend_text_style(legend, PlotConfig())

    assert legend.get_texts()[0].get_fontweight() == "bold"
    plt.close(fig)


def test_contour_contrast_color_uses_configured_threshold_and_colors() -> None:
    assert contour_contrast_color("viridis", PlotConfig(contour_luminance_threshold=0.0)) == "black"
    assert contour_contrast_color("viridis", PlotConfig(contour_luminance_threshold=1.0)) == "white"
    assert contour_contrast_color(
        "viridis",
        PlotConfig(contour_luminance_threshold=0.0, contour_dark_line_color="0.1"),
    ) == "0.1"
