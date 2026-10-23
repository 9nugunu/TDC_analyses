from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from deflector_tuning.visualization import s11_frequency_plots as s11
from deflector_tuning.visualization import phase_advance_plots as phase
from deflector_tuning.visualization import nodal_shift_plots as nodal
from deflector_tuning.visualization.polar_plot_plans import build_polar_plot_plans, PolarKyhlPairPlan
from deflector_tuning.visualization.polar_overlays import _draw_kyhl_phase_pair_overlay, _draw_family_overlay
from deflector_tuning.visualization.plot_config import PlotConfig


def test_s11_retains_arbitrary_geometry_and_draws_magnitude_only(tmp_path, monkeypatch):
    table = pd.DataFrame([dict(source_file='shared.txt', tune_position=.5, freq_ghz=f,
                              s_db=-6+offset, s_phase_deg=np.nan,
                              sim_DepthPlunger_offset=offset)
                          for offset in (-3., 3.) for f in (2.8, 2.9)])
    collapsed = s11._collapse_unlabelled_duplicate_rows(table)
    assert len(collapsed) == 4
    assert collapsed.s_phase_deg.isna().all()
    figures = []
    monkeypatch.setattr(s11, 'save_figure', lambda fig, path, config: figures.append(fig) or path)
    s11._plot_one(collapsed, pd.DataFrame(), tmp_path/'s11.png', title='magnitude', config=PlotConfig())
    lines = figures[0].axes[0].lines
    assert len(lines) == 2
    assert {tuple(line.get_ydata()) for line in lines} == {(-9., -9.), (-3., -3.)}
    assert all('sim_DepthPlunger_offset=' in line.get_label() for line in lines)


def _polar_rows():
    return pd.DataFrame([dict(source_file=f'{port}_{pos}.s1p', tune_position=pos,
                             marker_name='f_2pi3', s_phase_deg=angle, port_side=port)
                         for port, angles in [('in', (10., 30.)), ('out', (100., 170.))]
                         for pos, angle in zip((.5, 1.5), angles)])


def test_polar_two_ports_preserve_rotations_independent_of_row_order(tmp_path):
    results = []
    for table in (_polar_rows(), _polar_rows().iloc[::-1]):
        result = {}
        for plan in build_polar_plot_plans(table, tmp_path):
            if not isinstance(plan, PolarKyhlPairPlan):
                continue
            fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
            _draw_kyhl_phase_pair_overlay(ax, plan.pair_table, plan.title, plan.pair_label, config=PlotConfig())
            arcs = [line for line in ax.lines if line.get_gid() == 'kyhl_phase_rotation_arc']
            assert len(arcs) == 1
            rotation = np.rad2deg(arcs[0].get_xdata()[-1] - arcs[0].get_xdata()[0])
            port = plan.pair_table.port_side.iloc[0]
            assert f'port_side={port}' in ax.get_title()
            result[port] = rotation
            plt.close(fig)
        results.append(result)
    assert results[0] == pytest.approx({'in':20., 'out':70.})
    assert results[1] == pytest.approx(results[0])


def test_polar_ambiguous_sources_raise_instead_of_last_row_wins():
    table = _polar_rows().drop(columns='port_side')
    for ordered in (table, table.iloc[::-1]):
        fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
        with pytest.raises(ValueError, match='Ambiguous polar overlay'):
            _draw_family_overlay(ax, ordered, 'ambiguous', config=PlotConfig())
        plt.close(fig)


def _transition_rows():
    return pd.DataFrame([dict(dataset_id='demo', marker_name='f_2pi3', pos_from=pos, pos_to=pos+1,
                             phase_adv_deg=angle, phase_err_240_deg=angle-240,
                             phase_err_deg=angle-240, phase_err_abs_deg=abs(angle-240), phase_target_deg=240.,
                             sim_DepthPlunger_offset=offset, port_side=port)
                         for offset, angle in [(-3.,220.), (3.,260.)]
                         for port in ('in','out') for pos in (.5,1.5,2.5)])


def test_phase_facets_keep_arbitrary_geometry_out_of_same_curve(tmp_path, monkeypatch):
    figures=[]
    monkeypatch.setattr(phase, 'save_figure', lambda fig, path, config: figures.append(fig) or path)
    paths=phase.plot_phase_advance(_transition_rows(), tmp_path, split_by_family=True)
    assert len(paths)==2
    for fig in figures:
        ax=fig.axes[0]
        data=[line for line in ax.lines if line.get_label()!='ideal 240°']
        assert len(data)==2
        assert all(len(line.get_xdata())==3 for line in data)
        assert all(len(set(line.get_ydata()))==1 for line in data)
        assert 'sim_DepthPlunger_offset=' in ax.get_title()


def test_nodal_separates_geometry_ports_and_preserves_cumulative_values(tmp_path, monkeypatch):
    captured=[]
    monkeypatch.setattr(nodal,'save_figure',lambda fig,path,config: captured.append((path,fig)) or path)
    nodal.plot_nodal_shift(_transition_rows(),tmp_path)
    cumulative=[(p,fig.axes[0]) for p,fig in captured if p.stem.endswith('cumulative_phase')]
    assert len(cumulative)==4
    measured=[]
    for path,ax in cumulative:
        line=next(line for line in ax.lines if line.get_label()=='Measured')
        measured.append(float(line.get_ydata()[0]))
        assert 'sim_DepthPlunger_offset=' in ax.get_title()
        assert 'port_side=' in ax.get_title()
    assert sorted(measured)==[220.,220.,260.,260.]


def test_nodal_grid_mean_disclosed_in_title_and_legend(tmp_path, monkeypatch):
    table = _transition_rows().query("port_side == 'in'").drop(columns=['sim_DepthPlunger_offset'])
    table['sim_r_c']=[1.,1.,1.,2.,2.,2.]
    table['sim_w_c']=3.
    table=nodal._prepare_table(table)
    figures=[]
    monkeypatch.setattr(nodal,'save_figure',lambda fig,path,config: figures.append(fig) or path)
    nodal._plot_marker_cumulative_phase(table,tmp_path/'mean.png',marker='f_2pi3',family='cell',config=PlotConfig())
    ax=figures[0].axes[0]
    assert 'Mean over 2 conditions: sim_r_c' in ax.get_title()
    line=next(line for line in ax.lines if line.get_label()=='Mean measured')
    assert list(line.get_ydata())==[240.,480.,720.]

@pytest.mark.parametrize('with_phase_column', [True, False])
def test_public_s11_accepts_magnitude_only_markers(tmp_path, monkeypatch, with_phase_column):
    table = pd.DataFrame({'source_file':['trace.txt']*2, 'freq_ghz':[2.8,2.9], 's_db':[-6.,-8.]})
    markers = pd.DataFrame({'source_file':['trace.txt'], 'freq_ghz':[2.85], 's_db':[-7.], 'marker_name':['f_2pi3']})
    if with_phase_column:
        table['s_phase_deg']=np.nan
        markers['s_phase_deg']=np.nan
    figures=[]
    monkeypatch.setattr(s11,'save_figure',lambda fig,path,config: figures.append(fig) or path)
    table['tune_position']=.5
    markers['tune_position']=.5
    paths=s11.plot_s11_with_markers(table,markers,tmp_path,split_by_position=True)
    assert paths
    ax=figures[0].axes[0]
    assert list(ax.lines[0].get_ydata())==[-6.,-8.]
    annotations=[text.get_text() for text in ax.texts]
    assert annotations
    assert all('nan' not in text.lower() and '°' not in text for text in annotations)


def test_depth_state_preserves_phase_transitions_and_polar_pairs(tmp_path):
    from deflector_tuning.analysis.phase_advance import compute_phase_advance
    from deflector_tuning.data_contracts import simulation_geometry_columns
    from deflector_tuning.visualization.simulation_grouping import varying_sim_sweep_columns
    rows=_polar_rows().query("port_side == 'in'").copy()
    for column,value in {'dataset_id':'depth','data_kind':'sim','data_layer':'sim','marker_role':'sim',
                         's_name':'S11','freq_target_ghz':2.856,'freq_ghz':2.856,'s_db':-1.}.items():
        rows[column]=value
    rows['sim_NumDepth']=[1,2]
    rows['sim_DepthPlunger']=[30.,40.]
    rows['sim_DepthPlunger_offset']=3.
    assert simulation_geometry_columns(rows)==['sim_DepthPlunger_offset']
    assert varying_sim_sweep_columns(rows)==[]
    result=compute_phase_advance(rows)
    assert len(result)==1
    assert result.iloc[0].pos_from==.5 and result.iloc[0].pos_to==1.5
    assert result.iloc[0].sim_DepthPlunger_offset==3.
    plans=build_polar_plot_plans(rows,tmp_path)
    assert len([p for p in plans if isinstance(p,PolarKyhlPairPlan)])==1


def test_full_runner_renders_magnitude_only_s11_without_phase_products(tmp_path):
    from deflector_tuning.runner import run_folder_analysis
    folder=tmp_path/'data'/'sim'/'sim_grid_undated_magnitude'
    folder.mkdir(parents=True)
    (folder/'trace.txt').write_text(
        '#Parameters = {NumTune=0; r_c=52.09; w_c=19.3224}\n'
        '#"Frequency / GHz"\t"S1,1 (20) [Magnitude]"\n'
        '#-----------------------------------------\n'
        '2.8\t-6\n2.9\t-8\n',encoding='utf-8')
    dispersion=tmp_path/'data'/'sim'/'sim_dispersion_undated_reference'
    (dispersion/'processed').mkdir(parents=True)
    (dispersion/'processed'/'dispersion_summary.csv').write_text(
        'mode_index,freq_90_GHz,freq_120_GHz\n1,2.88,2.85\n',encoding='utf-8')
    result=run_folder_analysis(sparameter_path=folder,dispersion_path=dispersion,
                               output_dir=tmp_path/'output',marker_role='sim')
    assert result.figures['s11']
    assert all(path.exists() for path in result.figures['s11'].values())
    assert not {'polar','phase_advance','nodal_shift'}.intersection(result.analysis_modes)
