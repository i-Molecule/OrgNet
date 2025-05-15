from typing import Optional, List, Tuple, Dict, Any, Literal, Union
from pathlib import Path
import numpy as np
from copy import deepcopy
import py3Dmol


def _clamp_color(x: int) -> int: 
    return max(0, min(x, 255))
    
def get_hex_string_from_rgb(color: np.ndarray) -> str:
    if not isinstance(color, np.ndarray):
        color = np.array(color)
    color = (color * 255).astype(int).tolist()[:3]
    color = "#{0:02x}{1:02x}{2:02x}".format(
        _clamp_color(color[0]), _clamp_color(color[1]), _clamp_color(color[2]))
    return color

def convert_colors(values: np.ndarray, color : np.ndarray = [1, 0, 0]) -> np.ndarray:
    color = np.array(color, np.float32)
    return (1. - np.expand_dims(1. - color, 0) * np.expand_dims(values, 1)).tolist()


class StructureVisualizer:
    def __init__(self, window_size : Tuple[int, int] = (800, 600)):
        self.window_size = window_size
        self.init_view()
        
    def init_view(self):
        self.view = py3Dmol.view(width=self.window_size[0], height=self.window_size[1])
        self.clear()

    def clear(self):
        self.view.removeAllModels()
        self.view.removeAllShapes()
        self.view.removeAllSurfaces()
        self.view.removeAllLabels()
        self._current_model = 0
        self._model_names : List[str] = []
        self._model_styles : List[List[Dict[str, Any]]] = []
        self._created_shapes = []
        self._created_labels = []
        self._created_surfaces = []
        return
    
    def show(self):
        self.view.zoomTo()
        self.view.show()
        
    def _add_model(self,
            data: str,
            format : str = "pdb",
            empty_style : bool = True,
            name : Optional[str] = None,
            ) -> int:
        self.view.addModel(data, format)
        if empty_style:
            self.view.setStyle({"model": self._current_model}, {})
        self._model_styles.append([])
        if name is None:
            name = f"{self._current_model}"
        self._model_names.append(name)
        self._current_model += 1
        return self._current_model - 1
    
    @staticmethod
    def _get_style_dict_cartoon(color : str = "spectrum", **kwargs) -> Dict[str, Any]:
        kwargs.update({"color": color})
        return {"cartoon": kwargs}
    
    @staticmethod
    def _get_style_dict_sphere(
            color : Union[str, List[float]] = [1., 0., 0.],
            radius : float = 1.,
            opacity : float = 1.,
            **kwargs,
            ) -> Dict[str, Any]:
        if not isinstance(color, str):
            color = get_hex_string_from_rgb(color)
        kwargs.update({
            "color": color,
            "radius": float(radius),
            "opacity": float(opacity),
        })
        return {"sphere": kwargs}
    
    @staticmethod
    def _get_style_dict_stick(
            colorscheme : str = "whiteCarbon", 
            color : Optional[Union[str, List[float]]] = None,
            **kwargs) -> Dict[str, Any]:
        if color is None:
            kwargs.update({"colorscheme": colorscheme})
        else:
            if not isinstance(color, str):
                color = get_hex_string_from_rgb(color)
            kwargs.update({"color": color})
        return {"stick": kwargs}
    
    @classmethod
    def _get_style_dict(cls, mode : Literal["cartoon", "sphere", "stick"], **kwargs) -> Dict[str, Any]:
        if mode == "cartoon":
            return cls._get_style_dict_cartoon(**kwargs)
        elif mode == "sphere":
            return cls._get_style_dict_sphere(**kwargs)
        elif mode == "stick":
            return cls._get_style_dict_stick(**kwargs)
        else:
            raise ValueError(f"Unknown style: {mode}")
    
    def _add_model_style(self, model: int, style: Literal["cartoon", "sphere", "stick"], **kwargs):
        style = self._get_style_dict(style, **kwargs)
        self._model_styles[model].append(style)
        self.view.addStyle({"model": model}, style)
        return
    
    def _add_surface(self,
            atomsel: Dict[str, Any],
            allsel : Optional[Dict[str, Any]] = None,
            surface_type : Literal["VDW", "MS", "SAS", "SES"] = "MS",
            color : Union[str, List[float]] = "white",
            opacity : float = 1.,
            **kwargs,
            ):
        surface_type_mapping = {
            "VDW": py3Dmol.VDW,
            "MS": py3Dmol.MS,
            "SAS": py3Dmol.SAS,
            "SES": py3Dmol.SES,
        }
        surface_type = surface_type_mapping[surface_type.upper()]
        if not isinstance(color, str):
            color = get_hex_string_from_rgb(color)
        kwargs.update({
            "color": color,
            "opacity": opacity,
        })
        if allsel is None:
            obj = self.view.addSurface(surface_type, kwargs, atomsel)
        else:
            obj = self.view.addSurface(surface_type, kwargs, atomsel, allsel)
        self._created_surfaces.append(obj)
        return
    
    def _add_label(self,
            text: Union[str, List[str]],
            position : Optional[np.ndarray] = None,
            backgroundColor : str = "lightgray",
            backgroundOpacity : float = 0.8,
            fontColor : str = "black",
            borderThickness : int = 0,
            fontSize : int = 18,
            inFront : bool = True,
            offset : float = 1.5,
            selection : Optional[Dict[str, Any]] = None,
            **kwargs,
            ):
        if backgroundColor is not None and not isinstance(backgroundColor, str):
            backgroundColor = get_hex_string_from_rgb(backgroundColor)
        if fontColor is not None and not isinstance(fontColor, str):
            fontColor = get_hex_string_from_rgb(fontColor)
        kwargs.update({
            "backgroundOpacity": float(backgroundOpacity),
            "backgroundColor": backgroundColor,
            "fontColor": fontColor,
            "borderThickness": borderThickness,
            "fontSize": fontSize,
            "inFront": inFront,
        })
        if position is not None:
            kwargs.update({"position": {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            }})
        if isinstance(text, str):
            if selection is None:
                obj = self.view.addLabel(text, kwargs)
            else:
                obj = self.view.addLabel(text, kwargs, selection)
            self._created_labels.append(obj)
        else:
            for i, t in enumerate(text):
                k = deepcopy(kwargs)
                k["position"]["y"] -= i * offset
                obj = self.view.addLabel(t, k)
                self._created_labels.append(obj)
        return
    
    def _add_sphere(self, 
            center: np.ndarray, 
            radius : float = 1.,
            color : Union[str, List[float]] = "magenta",
            opacity : float = 1.,
            **kwargs,
            ):
        if not isinstance(color, str):
            color = get_hex_string_from_rgb(color)
        kwargs.update({
            "center": {
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
            },
            "radius": float(radius),
            "color": color,
            "opacity": float(opacity),
        })
        obj = self.view.addSphere(kwargs)
        self._created_shapes.append(obj)
        return
    
    def show_structure(self,
            structure: Union[str, Path],
            show_cartoon : bool = True,
            cartoon_color : str = "white",
            show_sticks : bool = False,
            sticks_colorscheme : str = "yellowCarbon",
            show_surface : bool = False,
            surface_opacity : bool = 0.8,
            surface_color : str = "white",
            ):
        if isinstance(structure, Path) \
                or isinstance(structure, str) and Path(structure).exists():
            pdb_data = open(structure).read()
        else:
            pdb_data = structure
        
        model = self._add_model(pdb_data, "pdb", name="target")
        if show_cartoon:
            self._add_model_style(model, "cartoon", color=cartoon_color)
        if show_sticks:
            self._add_model_style(model, "stick", colorscheme=sticks_colorscheme)
        if show_surface:
            self._add_surface({"model": model}, surface_type="SAS",
                color=surface_color, opacity=surface_opacity)
        return
    
    def show_sel_mutations(self, 
            residues: Dict[Tuple[str, str, str], List[str]], 
            model : int = 0,
            color : Tuple[float, float, float] = [1., 0., 0.],
            ):
        color = get_hex_string_from_rgb(color)
        for (chain, resi, wt), mut in residues.items():
            sel = {
                "model": model,
                "chain": chain,
                "resi": resi,
            }
            label = f"{chain} {resi} {wt}: " + " ".join(f"{mt}" for mt in mut)
            self.view.addStyle(sel, {"stick": {"color": color, "radius": 0.5}})
            self._add_label(label, selection=sel)
        return
    
    def show_sel_residue_scores(self,
            residues : Dict[Tuple[str, int, str], Dict[str, Tuple[float, float]]],
            model : int = 0,
            min_value : float = -1.,
            max_value : float = 1.,
            color_neg : Tuple[float, float, float] = [0., 0., 1.],
            color_pos : Tuple[float, float, float] = [1., 0., 0.],
            ):
        color_neg = np.array(color_neg)
        color_pos = np.array(color_pos)
        for (chain, resi, wt), mut_dict in residues.items():
            sel = {
                "model": model,
                "chain": chain,
                "resi": resi,
            }
            
            mean_value = np.mean([v[0] for v in mut_dict.values()])
            if mean_value > 0:
                color = (1. - (1. - color_pos) * mean_value / max_value)
                color = get_hex_string_from_rgb(color)
            elif mean_value < 0:
                color = (1. - (1. - color_neg) * (mean_value / min_value))
                color = get_hex_string_from_rgb(color)
            else:
                color = "#ffffff"
            
            label = f"{chain} {resi} {wt}: " + \
                " ".join(f"{mt}: {v1:.2f}±{v2:.2f}" for mt, (v1, v2) in mut_dict.items())
            
            self.view.addStyle(sel, {"stick": {"color": color, "radius": 0.5}})
            self._add_label(label, selection=sel)
        return
    