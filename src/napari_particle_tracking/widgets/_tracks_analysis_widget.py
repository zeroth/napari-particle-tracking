import warnings
from functools import partial
from typing import List, Optional, Tuple
from pathlib import Path

import napari.layers
from napari.utils import notifications
import napari.utils
import napari.utils.events
import numpy as np

from qtpy.QtCore import Signal
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QFileDialog
)

from napari_particle_tracking.libs import (
    basic_msd_fit,
    histogram,
    msd,
    msd_fit_function,
)

from ._napari_layers_widget import NPLayersWidget

from ._plots import create_histogram_widget, colors


class TracksAnaysisWidget(QWidget):
    trackSelected = Signal(int)
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        nplayers_widget: NPLayersWidget,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.viewer: napari.viewer.Viewer = viewer
        self._napari_layers_widget: NPLayersWidget = nplayers_widget
        self.setLayout(QFormLayout())

        self._timedelay = QDoubleSpinBox(self)
        self._timedelay.setMinimum(0)
        self._timedelay.setValue(5)
        self._timedelay.setSingleStep(1)
        self._timedelay.setSuffix(" ms")
        self.layout().addRow("Time Delay", self._timedelay)

        self._max_try = QLineEdit(self)
        self._max_try.setText("1000000")
        self._max_try.setValidator(QIntValidator())
        self.layout().addRow("Max Try for msd_fit", self._max_try)

        self._intensity_option = QComboBox(self)
        options = [
            {"text": "Max Intensity", "userData": "max_intensity"},
            {"text": "Mean Intensity", "userData": "mean_intensity"},
            {"text": "Min Intensity", "userData": "min_intensity"},
        ]
        for option in options:
            self._intensity_option.addItem(option["text"], option["userData"])
        self.layout().addRow("Intensity Option", self._intensity_option)

        self._btn_analyze = QPushButton("Analyze")
        self.layout().addRow(self._btn_analyze)
        self._btn_analyze.clicked.connect(self._analyze)

        self._btn_download = QPushButton("Download")
        self.layout().addRow(self._btn_download)
        self._btn_download.clicked.connect(self._download)

        self._plot_scroll = QScrollArea(self)
        self._plot_scroll.setWidgetResizable(True)
        self.layout().addRow(self._plot_scroll)
        self.selected_track = None

    def _download(self):
        # get the tracks layer
        _tracks_layer: napari.layers.Tracks = self._napari_layers_widget.get_selected_layers().get(
            "Tracks", None
        )

        if _tracks_layer is None:
            warnings.warn("Please select/add a Tracks layer.")
            return
        
        tracks_df = _tracks_layer.metadata["original_tracks_df"]

        save_path = QFileDialog.getExistingDirectory(self, "Save Data")
        if save_path:
            save_path = Path(save_path)
            priject_name = _tracks_layer.name.removeprefix("Tracks_")
            _all_tracks_path = save_path.joinpath(f"{priject_name}_all_tracks.csv")
            tracks_df.to_csv(_all_tracks_path, index=False)

            _msd_path = save_path.joinpath(f"{priject_name}_msd.csv")
            self.tracked_msd.to_csv(_msd_path)

            _msd_fit_path = save_path.joinpath(f"{priject_name}_msd_fit.csv")
            self.tracked_msd_fit.to_csv(_msd_fit_path)

            _filtered_tracks_path = save_path.joinpath(f"{priject_name}_filtered_tracks.csv")
            self.filtered_tracks_df.to_csv(_filtered_tracks_path)
            
            notifications.show_info(f"Data saved successfully at {save_path}")
        

    def _analyze(self):
        # get the tracks layer
        _tracks_layer: napari.layers.Tracks = self._napari_layers_widget.get_selected_layers().get(
            "Tracks", None
        )

        if _tracks_layer is None:
            warnings.warn("Please select/add a Tracks layer.")
            return

        notifications.show_info("Analyzing track, please wait...")

        _intensity_filter = self._intensity_option.currentData()
        _intensity_filter_type = self._intensity_option.currentText()

        # get tracks df from the layers metadata filter it to the current tracks

        tracks_df = _tracks_layer.metadata["original_tracks_df"]
        current_tracks = _tracks_layer.data
        current_track_ids = list(set(current_tracks[:, 0]))
        current_tracks_df = tracks_df[
            tracks_df["track_id"].isin(current_track_ids)
        ]
        self.filtered_tracks_df = current_tracks_df
        _tracks_layer.metadata["filtered_tracks_df"] = self.filtered_tracks_df

        # get track lengths
        track_lengths = current_tracks_df.groupby("track_id").size().to_numpy()

        # get track mean intensity
        track_mean_intensity = (
            current_tracks_df.groupby("track_id")[_intensity_filter]
            .mean()
            .to_numpy()
        )
        binsize = 100 if np.mean(track_mean_intensity) >= 1000 else 5

        # get track msd
        current_track_msd = None
        # check if the tracks have z
        if "z" in current_tracks_df.columns:
            # get msd for 3D todo have a corrent distance calculation
            current_track_msd = current_tracks_df.groupby(
                "track_id", group_keys=True
            ).apply(lambda x: msd(x[["x", "y", "z"]].to_numpy()))
        else:
            current_track_msd = current_tracks_df.groupby(
                "track_id", group_keys=True
            ).apply(lambda x: msd(x[["x", "y"]].to_numpy()))

        self.tracked_msd = current_track_msd
        # print("tracked_msd coloum: ", self.tracked_msd.to_frame().reset_index().columns)
        _tracks_layer.metadata["tracked_msd"] = self.tracked_msd.reset_index()
        _tracks_layer.metadata["msd_delta"] = float(self._timedelay.value())
        # fit the msd
        _basic_fit_partial = partial(
            basic_msd_fit,
            delta=float(self._timedelay.value()),
            fit_function=msd_fit_function,
            maxfev=int(self._max_try.text()),
        )

        current_track_fit_df_main = current_track_msd.groupby(
            "track_id", group_keys=True
        ).apply(lambda x: _basic_fit_partial(x.to_numpy()))

        current_track_fit_df = current_track_fit_df_main.groupby("track_id").first()
        current_track_fit_df = current_track_fit_df['alpha']
        current_track_fit = current_track_fit_df.to_numpy()
        
        self.tracked_msd_fit = current_track_fit_df_main
        _tracks_layer.metadata["tracked_msd_fit"] = self.tracked_msd_fit.reset_index()

        _confined_tracks = current_track_fit_df[
            current_track_fit_df < 0.4
        ].index.to_numpy()
        _diffusive_tracks = current_track_fit_df[
            (current_track_fit_df >= 0.4) & (current_track_fit_df <= 1.2)
        ].index.to_numpy()
        _directed_tracks = current_track_fit_df[
            current_track_fit_df > 1.2
        ].index.to_numpy()

        
        # get mean intensity for each category
        _mean_intensity_confined = (
            current_tracks_df[
                current_tracks_df["track_id"].isin(_confined_tracks)
            ]
            .groupby("track_id")[_intensity_filter]
            .mean()
            .to_numpy()
        )
        _mean_intensity_diffusive = (
            current_tracks_df[
                current_tracks_df["track_id"].isin(_diffusive_tracks)
            ]
            .groupby("track_id")[_intensity_filter]
            .mean()
            .to_numpy()
        )
        _mean_intensity_directed = (
            current_tracks_df[
                current_tracks_df["track_id"].isin(_directed_tracks)
            ]
            .groupby("track_id")[_intensity_filter]
            .mean()
            .to_numpy()
        )

        # print(f"Total Tracks: {len(track_lengths)}")
        # create dict to store parameters for histogram widget to be created
        _hist_params = [
            {
                "values": track_lengths,
                "binsize": 1,
                "xlabel": "Length",
                "ylabel": "Number of Tracks",
                "title": "Track Length Histogram",
                "histtype": "bar",
                "info": f"Total Tracks: {len(track_lengths)}",
            },
            {
                "values": track_mean_intensity,
                "binsize": binsize,
                "xlabel": _intensity_filter_type,
                "ylabel": "Number of Tracks",
                "title": f"Track {_intensity_filter_type} Histogram",
                "histtype": "bar",
                "info": f"Total Tracks: {len(track_mean_intensity)}",
            },
            {
                "values": current_track_fit,
                "binsize": 0.1,
                "xlabel": "MSD α",
                "ylabel": "Number of Tracks",
                "title": "Track MSD Fit",
                "histtype": "line",
                "legends": [
                    "Confined α < 0.4",
                    "Diffusive 0.4 < α < 1.2",
                    "Directed α > 1.2",
                ],
                "vspan": [
                    np.min(current_track_fit),
                    0.4,
                    1.2,
                    np.max(current_track_fit),
                ],
                "info": f"Total confiend: {len(_confined_tracks)}, diffusive: {len(_diffusive_tracks)}, directed: {len(_directed_tracks)}",
            },
            {
                "values": _mean_intensity_confined,
                "binsize": binsize,
                "xlabel": _intensity_filter_type,
                "ylabel": "Number of Tracks",
                "title": f"Track {_intensity_filter_type} Confined Histogram",
                "histtype": "bar",
                "info": f"Total Tracks: {len(_mean_intensity_confined)}",
            },
            {
                "values": _mean_intensity_diffusive,
                "binsize": binsize,
                "xlabel": _intensity_filter_type,
                "ylabel": "Number of Tracks",
                "title": f"Track {_intensity_filter_type} Diffusive Histogram",
                "histtype": "bar",
                "info": f"Total Tracks: {len(_mean_intensity_diffusive)}",
            },
            {
                "values": _mean_intensity_directed,
                "binsize": binsize,
                "xlabel": _intensity_filter_type,
                "ylabel": "Number of Tracks",
                "title": f"Track {_intensity_filter_type} Directed Histogram",
                "histtype": "bar",
                "info": f"Total Tracks: {len(_mean_intensity_directed)}",
            },
        ]

        # create histogram widgets
        _plot_widget = QWidget()
        _plot_widget.setLayout(QVBoxLayout())
        for i, _hist_param in enumerate(_hist_params):
            _hist_param["color"] = colors[i % len(colors)]
            _hist_plot_widget = create_histogram_widget(**_hist_param)
            _hist_plot_widget.setMinimumWidth(400)
            _hist_plot_widget.setMinimumHeight(400)
            _plot_widget.layout().addWidget(_hist_plot_widget)
            # self._hist_plot_widgets.append(_hist_plot_widget)

        # add eveything to the scroll area
        swid = self._plot_scroll.widget()
        if swid is not None:
            swid.deleteLater()

        self._plot_scroll.setWidget(_plot_widget)
        self.append_mouse_callback(_tracks_layer)
        notifications.show_info("Track analysis completed.")

    def append_mouse_callback(self, track_layer: napari.layers.Tracks) -> None:
        """
        Add a mouse callback to ``track_layer`` to draw the tree
        when the layer is clicked.
        """

        @track_layer.mouse_double_click_callbacks.append
        def show_track(tracks: napari.layers.Tracks, event: napari.utils.events.Event) -> None:
            self.tracks = tracks

            cursor_position = event.position
            track_id = tracks.get_value(cursor_position, world=True)
            if track_id is not None:
                self.trackSelected.emit(int(track_id))
