from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from deflector_tuning.visualization import grid_scan_phase_line_plots as phase
from deflector_tuning.visualization import grid_scan_spacing_maps as maps
from deflector_tuning.visualization import raw_y11_plots as raw
from deflector_tuning.visualization import em_field_structure_plots as field
from deflector_tuning.visualization.plot_config import PlotConfig, apply_plot_style
from deflector_tuning.visualization.lps_paper_style import apply_lps_style
from deflector_tuning.data_loading.field_profiles import FieldPhaseTrace


def capture_figure(monkeypatch, module):
    captured = []
    def save(fig, path, config):
        captured.append(fig)
        return Path(path)
    monkeypatch.setattr(module, 'save_figure', save)
    return captured


def test_field_phase_legend_follows_component_and_preserves_values():
    fig, ax = plt.subplots()
    traces = tuple(FieldPhaseTrace('raw', kind, component, np.array([0., 1.]), np.array([12., 34.]))
                   for kind, component in [('e', 'Y'), ('h', 'X')])
    field._draw_phase_traces(ax, traces, config=PlotConfig())
    assert [line.get_label() for line in ax.lines] == ['$E_y$ phase', '$H_x$ phase']
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [12., 34.])
    plt.close(fig)


def test_map_minimum_and_scatter_use_same_averages_and_axis_units(tmp_path, monkeypatch):
    captured = capture_figure(monkeypatch, maps)
    table = pd.DataFrame({'sim_r_c': [56., 56., 57.], 'sim_w_c': [20., 20., 20.], 'error': [0., 100., 10.]})
    maps._plot_grid_value_map(table, tmp_path/'map.png', 'sim_w_c', 'sim_r_c', 'error', 'error',
                             PlotConfig(math_bold=False), design_point=None, cmap='viridis', draw_best=True)
    ax = captured[0].axes[0]
    np.testing.assert_array_equal(ax.collections[0].get_array(), [50., 10.])
    np.testing.assert_array_equal(ax.collections[1].get_offsets(), [[20., 57.]])
    assert ax.get_xlabel() == '$w_c$ [mm]'
    assert ax.get_ylabel() == '$r_c$ [mm]'


def test_radius_filename_precision_preserves_both_outputs(tmp_path, monkeypatch):
    captured = capture_figure(monkeypatch, raw)
    table = pd.DataFrame([dict(sim_r_c=radius, source_file=f'{radius}.txt', freq_ghz=2.8,
                              y_re_siemens=radius, y_im_siemens=0.) for radius in [56.091, 56.094]])
    markers = pd.DataFrame(columns=['sim_r_c','marker_name','y_re_siemens','y_im_siemens'])
    outputs = raw.plot_y11_raw_frequency_with_markers(table, markers, tmp_path)
    assert len(set(outputs.values())) == 2
    assert [fig.axes[0].lines[0].get_ydata()[0] for fig in captured] == [56091., 56094.]


def test_paper_then_general_style_is_identical_to_general_from_defaults():
    with mpl.rc_context():
        mpl.rcdefaults()
        apply_plot_style()
        expected = dict(mpl.rcParams)
        apply_lps_style()
        apply_plot_style()
        assert dict(mpl.rcParams) == expected
        fig, ax = plt.subplots()
        ax.set_title('ordinary label')
        assert not ax.title.get_usetex()
        fig.canvas.draw()
        plt.close(fig)


def test_phase_width_config_selects_data_and_explicit_override_wins(tmp_path, monkeypatch):
    captured = capture_figure(monkeypatch, phase)
    rows = pd.DataFrame(dict(sim_r_c=[56.,56.], sim_w_c=[19.3224,20.], marker_name=['f_mean']*2, s_phase_deg=[12.,42.]))
    config = PlotConfig(design_point_by_axis={'sim_w_c':20.})
    phase.plot_grid_scan_sparameter_phase_r_c_line_scan(rows, tmp_path, config=config)
    phase.plot_grid_scan_sparameter_phase_r_c_line_scan(rows, tmp_path, config=config, fixed_w_c=19.3224)
    assert list(captured[0].axes[0].lines[0].get_ydata()) == [42.]
    assert list(captured[1].axes[0].lines[0].get_ydata()) == [12.]
    assert pd.read_csv(tmp_path/'sparameter_phase_vs_r_c_at_w_c_20.csv').sim_w_c.tolist() == [20.]


def test_candidate_cli_labels_reach_actual_annotations(tmp_path, monkeypatch):
    from scripts import plot_rc_candidate_on_phase_line as cli
    captured = capture_figure(monkeypatch, phase)
    source = tmp_path/'line.csv'
    pd.DataFrame(dict(sim_r_c=[56.,57.], marker_name=['f_mean']*2, s_phase_deg=[12.,42.])).to_csv(source,index=False)
    monkeypatch.setattr(sys,'argv',['plot',str(source),str(tmp_path/'out.png'),
        '--candidate-r-c-mm','56.3','--candidate-label','Candidate A',
        '--before-r-c-mm','56.1','--before-label','Initial A',
        '--target-r-c-mm','56.8','--target-label','Target A'])
    cli.main()
    texts = {text.get_text() for text in captured[0].axes[0].texts}
    assert {'Candidate A','Initial A','Target A'} <= texts


def test_generic_state_labels_and_legacy_alias(tmp_path, monkeypatch):
    captured = capture_figure(monkeypatch, phase)
    table = pd.DataFrame(dict(sim_r_c=[56.,57.], marker_name=['f_mean']*2, s_phase_deg=[12.,42.]))
    common = dict(before_r_c_mm=56.1,current_r_c_mm=56.2,design_r_c_mm=56.7,f_mean_zero_r_c_mm=56.5)
    phase.plot_phase_rc_states(table,tmp_path/'new.png',additional_r_c_mm=56.3,additional_label='After tuning',current_label='Measured',**common)
    phase.plot_phase_rc_states(table,tmp_path/'old.png',s003_r_c_mm=56.3,**common)
    assert {'After tuning','Measured'} <= {text.get_text() for text in captured[0].axes[0].texts}
    assert all('s002' not in text.get_text() and 's003' not in text.get_text() for text in captured[1].axes[0].texts)
    assert any(np.all(np.asarray(line.get_xdata()) == 56.3) for line in captured[1].axes[0].lines)


def test_advance_bar_title_uses_actual_positions(tmp_path, monkeypatch):
    from deflector_tuning.visualization import position_phase_bar_plots as bars
    captured = capture_figure(monkeypatch, bars)
    rows = pd.DataFrame([dict(dataset_id='sample',source_file='sample.s1p',tune_position=position,
                             marker_name=marker,freq_target_ghz=2.8,freq_ghz=2.8,s_phase_deg=phase_value)
                         for position,phase_value in [(1.25,10.),(3.75,30.)]
                         for marker in ['f_2pi3','f_mean','f_pi2']])
    targets={position:{marker:angle for marker in ['f_2pi3','f_mean','f_pi2']}
             for position,angle in [(1.25,0.),(3.75,60.)]}
    bars.plot_position_phase_advance_bars(rows,tmp_path/'bars.png',positions=[1.25,3.75],ideal_phase_targets_deg=targets)
    assert captured[0].axes[0].get_title() == 'Phase advance: position 1.25 to 3.75'


def test_spacing_sidecar_uses_configured_width(tmp_path, monkeypatch):
    capture_figure(monkeypatch, maps)
    table=pd.DataFrame(dict(sim_r_c=[56.,57.],sim_w_c=[20.,20.],
                            spacing_60deg_target_error_deg=[1.,2.],spacing_equality_error_deg=[2.,3.]))
    maps.plot_grid_scan_spacing_error_maps(table,tmp_path,config=PlotConfig(design_point_by_axis={'sim_w_c':20.}))
    result=pd.read_csv(tmp_path/'grid_scan_spacing_r_c_line_scan_w_c_20.csv')
    assert result.sim_w_c.tolist() == [20.,20.]


def test_map_rejects_unknown_physical_axes(tmp_path):
    table=pd.DataFrame(dict(a=[1.,2.],b=[3.,4.],value=[1.,2.]))
    with pytest.raises(ValueError,match='distinct sim_r_c and sim_w_c'):
        maps._plot_grid_value_map(table,tmp_path/'bad.png','a','b','value','value',PlotConfig(),
                                 design_point=None,cmap='viridis',draw_best=True)


@pytest.mark.parametrize('design_point', [{}, {'sim_r_c': 56.5}])
def test_partial_design_mapping_preserves_default_scan_width_and_sidecars(tmp_path, monkeypatch, design_point):
    from deflector_tuning.project_defaults import DEFAULT_PROJECT_DEFAULTS
    from deflector_tuning.analysis.grid_rc_line_scan import rc_line_scan_filename
    width = DEFAULT_PROJECT_DEFAULTS.design_point_by_axis['sim_w_c']
    config = PlotConfig(design_point_by_axis=design_point)
    captured = capture_figure(monkeypatch, phase)
    capture_figure(monkeypatch, maps)
    table = pd.DataFrame(dict(sim_r_c=[56.,57.],sim_w_c=[width,width],marker_name=['f_mean']*2,
                             s_phase_deg=[10.,20.],spacing_60deg_target_error_deg=[1.,2.],
                             spacing_equality_error_deg=[2.,3.], dphase_d_sim_r_c_deg_per_mm=[1.,2.],
                             dphase_d_sim_w_c_deg_per_mm=[2.,3.], gradient_magnitude_deg_per_mm=[3.,4.]))
    outputs = phase.plot_grid_scan_sparameter_phase_r_c_line_scan(table,tmp_path,config=config)
    assert pd.read_csv(next(iter(outputs.values())).with_suffix('.csv')).sim_w_c.tolist() == [width,width]
    assert list(captured[0].axes[0].lines[0].get_ydata()) == [10.,20.]
    fit = pd.DataFrame(dict(state=['baseline','current','design'],r_c_mm=[56.1,56.2,56.7]))
    phase.plot_phase_rc_map(table,fit,tmp_path/'map.png',config=config)
    phase.plot_phase_rc_states(table,tmp_path/'states.png',before_r_c_mm=56.1,current_r_c_mm=56.2,
                              design_r_c_mm=56.7,f_mean_zero_r_c_mm=56.5,config=config)
    maps.plot_grid_scan_spacing_error_maps(table,tmp_path,config=config)
    maps.plot_grid_scan_phase_sensitivity_maps(table,tmp_path,config=config)
    for stem in ['grid_scan_spacing','grid_scan_phase_sensitivity']:
        assert pd.read_csv(tmp_path/rc_line_scan_filename(stem,width)).sim_w_c.tolist() == [width,width]


@pytest.mark.parametrize('state_label', [None, 'After alignment'])
def test_raw_phase_position_uses_measured_state_label(tmp_path, monkeypatch, state_label):
    from deflector_tuning.visualization import phase_radius_equivalence_plots as equivalence
    captured = capture_figure(monkeypatch, equivalence)
    position = pd.DataFrame([dict(marker_name='f_mean',before_phase_deg=10.,after_phase_deg=20.,
                                 marker_equivalent_r_c_mm=56.2,shared_equivalent_r_c_mm=56.2,
                                 remaining_to_target_um=300.)])
    curves = pd.DataFrame(dict(marker_name=['f_mean']*2,sim_r_c=[56.,57.],raw_anchored_phase_deg=[10.,30.]))
    observation = pd.DataFrame([dict(marker_name='f_mean',before_phase_deg=3.,after_phase_deg=8.,phase_delta_deg=5.)])
    kwargs = {} if state_label is None else {'measured_state_label':state_label}
    equivalence.plot_raw_phase_position_on_simulation_line(position,curves,observation,tmp_path/'state.png',
                                                          baseline_r_c_mm=56.,target_r_c_mm=56.5,
                                                          fixed_w_c_mm=20.,**kwargs)
    expected = state_label or 'Measured state'
    fig = captured[0]
    assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == ['before',expected]
    assert f'raw {expected}' in fig.axes[1].get_legend_handles_labels()[1]
    assert fig._suptitle.get_text().startswith(expected + ' located')
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(),[3.,8.])
    np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(),[10.,30.])
