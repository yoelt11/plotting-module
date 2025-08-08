import matplotlib.pyplot as plt
from typing import List, Optional
import jax.numpy as jnp
from . import structures

def set_style(mode="dark"):
    if mode == "dark":
        plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
    else:
        plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')


def curves_to_curve_with_std(curves: List[structures.Curve]) -> structures.CurveWithStd:
    mean = jnp.array([curve.values for curve in curves]).mean(axis=0)
    std = jnp.array([curve.values for curve in curves]).std(axis=0)
    return structures.CurveWithStd(mean, std, curves[0].items, curves[0].legend)


def plot_curves(curves: List[structures.Curve], title: Optional[str] = None,
                mode: str = "dark", ax: Optional[plt.Axes] = None) -> plt.Axes:
    set_style(mode)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    if title is not None:
        ax.set_title(title)

    for curve in curves:
        ax.plot(curve.items, curve.values, label=curve.legend)
    
    ax.set_yscale('log')
    ax.legend()
    
    return ax

def plot_curve_with_std(curves_with_std: List[structures.CurveWithStd], title: Optional[str] = None,
                       mode: str = "dark", ax: Optional[plt.Axes] = None) -> plt.Axes:
    set_style(mode)

    if ax is None:
        fig, ax = plt.subplots()
    
    if title is not None:
        ax.set_title(title)
    for curve_with_std in curves_with_std:
        ax.plot(curve_with_std.items, curve_with_std.mean, label=curve_with_std.legend)
        ax.fill_between(curve_with_std.items, curve_with_std.mean - curve_with_std.std, curve_with_std.mean + curve_with_std.std, alpha=0.2)
    ax.legend()
    return ax

def plot_single_solution(solution: structures.Solution, title: Optional[str] = None,
                         mode: str = "dark", ax: Optional[plt.Axes] = None) -> plt.Axes:
    set_style(mode)

    if ax is None:
        fig, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)

    if "Error" in title:
        cmap = "inferno"
    else:
        cmap = "jet"
    contour = ax.contourf(solution.x, solution.y, solution.values, levels=100, cmap=cmap)
    plt.colorbar(contour, ax=ax)
 
    return ax
     
def plot_multiple_solutions(item_1: List[structures.Solution], item_2: List[structures.Solution], item_3: List[structures.Solution], title: Optional[str] = None,
                            columnwise_titles: Optional[List[str]] = None,
                            mode: str = "dark") -> plt.Axes:
    set_style(mode)

    fig, axs = plt.subplots(len(item_1), 3, figsize=(len(item_1)*10, 3))

    if len(axs.shape) == 1:
        axs = axs.reshape(1,-3)
    if columnwise_titles is None:
        for i in range(len(item_1)):
            plot_single_solution(item_1[i], title=f"Prediction {i*3 + 0}", ax=axs[i, 0])
            plot_single_solution(item_2[i], title=f"Prediction {i*3 + 1}", ax=axs[i, 1])
            plot_single_solution(item_3[i], title=f"Prediction {i*3 + 2}", ax=axs[i, 2])
    else:
        for i in range(len(item_1)):
            plot_single_solution(item_1[i], title=columnwise_titles[0], ax=axs[i, 0])
            plot_single_solution(item_2[i], title=columnwise_titles[1], ax=axs[i, 1])
            plot_single_solution(item_3[i], title=columnwise_titles[2], ax=axs[i, 2])

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # allow space for suptitle

    return fig