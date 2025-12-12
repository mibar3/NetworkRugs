import matplotlib.pyplot as plt
import networkx as nx
import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ba_utils.orderings as orderings
import ba_utils.visualization as visualization

from load_real_data import load_real_network

# --- Global hover/click state ---
hover_timer = None
last_hover_coords = (None, None)
hover_preview_canvas = None
hover_delay_ms = 500
click_popups = []

# ---------------- Hover & Click Functions ----------------

def enable_hover(ax, fig, graphs_data, ordering, color_mapping, pixel_size=40):
    global hover_timer, last_hover_coords, hover_preview_canvas
    canvas = fig.canvas
    parent_widget = canvas.get_tk_widget().master

    def open_hover_preview(timestamp, node_id, x_canvas, y_canvas):
        global hover_preview_canvas
        if hover_preview_canvas:
            hover_preview_canvas.get_tk_widget().destroy()
            hover_preview_canvas = None
        fig_preview = plt.Figure(figsize=(2,2))
        ax_preview = fig_preview.add_subplot(111)
        G = graphs_data[timestamp]
        pos = nx.spring_layout(G)
        node_colors = [color_mapping.get((timestamp,n),'C0') for n in G.nodes()]
        nx.draw(G, pos, with_labels=False, node_size=50, font_size=4, node_color=node_colors, ax=ax_preview)
        ax_preview.axis('off')
        ax_preview.set_title(f"t={timestamp}", fontsize=8)
        hover_preview_canvas = FigureCanvasTkAgg(fig_preview, master=parent_widget)
        hover_preview_canvas.draw()
        hover_preview_canvas.get_tk_widget().place(x=x_canvas, y=y_canvas)

    def on_hover(event):
        global hover_timer, last_hover_coords, hover_preview_canvas
        if event.inaxes is None or not hasattr(event,'xdata') or not hasattr(event,'ydata'):
            return
        x, y = int(event.xdata), int(event.ydata)
        canvas_x, canvas_y = event.guiEvent.x, event.guiEvent.y

        def trigger_hover_preview():
            try:
                t_idx = x // pixel_size
                y_idx = y // pixel_size
                timestamps = sorted(graphs_data.keys())
                if t_idx >= len(timestamps): return
                timestamp = timestamps[t_idx]
                node_ordering = ordering[timestamp]
                if y_idx >= len(node_ordering): return
                node_id = node_ordering[y_idx]
                open_hover_preview(timestamp, node_id, canvas_x, canvas_y)
            except Exception as e:
                print("Error during hover preview:", e)

        last_x, last_y = last_hover_coords
        if last_x is not None and last_y is not None:
            move_distance = ((x-last_x)**2 + (y-last_y)**2)**0.5
            if move_distance > 5:
                if hover_timer:
                    parent_widget.after_cancel(hover_timer)
                hover_timer = None
                if hover_preview_canvas:
                    widget = hover_preview_canvas.get_tk_widget()
                    if widget.winfo_exists():
                        try: widget.destroy()
                        except tk.TclError: pass
                    hover_preview_canvas = None

        last_hover_coords = (x, y)
        if hover_timer is None:
            hover_timer = parent_widget.after(hover_delay_ms, trigger_hover_preview)

    def on_leave(event):
        global hover_preview_canvas, hover_timer
        if hover_timer:
            parent_widget.after_cancel(hover_timer)
            hover_timer = None
        if hover_preview_canvas:
            widget = hover_preview_canvas.get_tk_widget()
            if widget.winfo_exists():
                try: widget.destroy()
                except tk.TclError: pass
            hover_preview_canvas = None

    canvas.mpl_connect('motion_notify_event', on_hover)
    canvas.mpl_connect('figure_leave_event', on_leave)

def open_popup(timestamp, node_id, graphs_data, color_mapping):
    fig_graph = plt.Figure(figsize=(4,4))
    ax_graph = fig_graph.add_subplot(111)
    G = graphs_data[timestamp]
    pos = nx.spring_layout(G)
    node_colors = [color_mapping.get((timestamp,n),'C0') for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=200, font_size=8, font_color='white', node_color=node_colors, ax=ax_graph)
    ax_graph.set_title(f"Graph at t={timestamp}")

    popup = tk.Toplevel()
    popup.title(f"Graph at t={timestamp}")
    canvas_graph = FigureCanvasTkAgg(fig_graph, master=popup)
    canvas_graph.draw()
    canvas_graph.get_tk_widget().pack(fill='both', expand=True)
    click_popups.append(popup)

def close_all_popups():
    global click_popups
    for popup in click_popups:
        try: popup.destroy()
        except Exception: pass
    click_popups = []

def enable_click(ax, fig, graphs_data, ordering, color_mapping, pixel_size=40):
    canvas = fig.canvas
    def on_click(event):
        if event.inaxes is None or not hasattr(event,'xdata') or not hasattr(event,'ydata'): return
        x, y = int(event.xdata), int(event.ydata)
        try:
            t_idx = x // pixel_size
            y_idx = y // pixel_size
            timestamps = sorted(graphs_data.keys())
            if t_idx >= len(timestamps): return
            timestamp = timestamps[t_idx]
            node_ordering = ordering[timestamp]
            if y_idx >= len(node_ordering): return
            node_id = node_ordering[y_idx]
            open_popup(timestamp, node_id, graphs_data, color_mapping)
        except Exception as e: print("Error during click:", e)
    canvas.mpl_connect('button_press_event', on_click)

def create_tooltip(widget, text):
    tooltip = None
    def show_tooltip(event):
        nonlocal tooltip
        x, y, _, _ = widget.bbox("insert")
        x += widget.winfo_rootx() + 25
        y += widget.winfo_rooty() + 25
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(tooltip, text=text, justify='left', background="#ffffe0",
                          relief='solid', borderwidth=1, padding=(5,2,5,2))
        label.pack()
    def hide_tooltip(event):
        nonlocal tooltip
        if tooltip: tooltip.destroy(); tooltip=None
    widget.bind('<Enter>', show_tooltip)
    widget.bind('<Leave>', hide_tooltip)

# ---------------- Load JSON Data ----------------
graphs_data = {}


def load_and_open(filename):
    global graphs_data
    try:
        graphs_data = load_real_network(filename)
    except Exception as e:
        print("Error loading network:", e)
        return
    open_rug_window()

# ---------------- Rug Window ----------------
def open_rug_window():
    global graphs_data
    if not graphs_data:
        print("No graphs loaded yet.")
        return

    rug_window = tk.Toplevel()
    rug_window.title("NetworkRug Viewer")
    rug_window.geometry("1600x900")

    def on_rug_window_close():
        close_all_popups()
        rug_window.destroy()
    rug_window.protocol("WM_DELETE_WINDOW", on_rug_window_close)

    color_options = ['id','degree_centrality','closeness_centrality','betweenness_centrality','eigenvector_centrality']
    ordering_options = ['priority','priority_tunable','dfs','bfs','metric_ordering']

    color_var = tk.StringVar(value='id')
    order_var = tk.StringVar(value='priority')
    label_var = tk.BooleanVar()
    parameters_var = tk.StringVar(value="w=1,d=1,c=1")
    metric_var = tk.StringVar(value='degree')

    control_frame = ttk.Frame(rug_window)
    control_frame.pack(side='top', fill='x', pady=10)
    ttk.Label(control_frame, text="Color Encoding").grid(row=0,column=0,padx=5)
    ttk.Combobox(control_frame, textvariable=color_var, values=color_options).grid(row=0,column=1,padx=5)
    ttk.Label(control_frame, text="Ordering").grid(row=0,column=2,padx=5)
    ttk.Combobox(control_frame, textvariable=order_var, values=ordering_options).grid(row=0,column=3,padx=5)
    ttk.Checkbutton(control_frame, text="Show Labels", variable=label_var).grid(row=0,column=4,padx=5)

    rug_canvas = ttk.Frame(rug_window)
    rug_canvas.pack(fill='both', expand=True)

    def render_rug():
        fig = plt.Figure()
        ax = fig.add_subplot(111)

        # --- Ordering selection ---
        start_nodes = None
        if order_var.get() == "priority":
            ordering = orderings.get_priority_bfs_ordering(graphs_data, start_nodes)
        elif order_var.get() == "priority_tunable":
            param_dict = dict(item.split('=') for item in parameters_var.get().split(','))
            w=float(param_dict.get('w',1.0)); d=float(param_dict.get('d',1.0)); c=float(param_dict.get('c',1.0))
            ordering = orderings.get_tunable_priority_bfs_ordering(graphs_data, start_nodes,w=w,d=d,c=c)
        elif order_var.get() == "bfs":
            ordering = orderings.get_BFS_ordering(graphs_data, start_nodes)
        elif order_var.get() == "dfs":
            ordering = orderings.get_DFS_ordering(graphs_data, start_nodes)
        elif order_var.get() == "metric_ordering":
            ordering = orderings.get_centrality_ordering(graphs_data, metric_var.get())
        else:
            ordering = orderings.get_priority_bfs_ordering(graphs_data, start_nodes)

        # --- Draw NetworkRug ---
        fig, color_mapping = visualization.draw_rug_with_color_mapping(
            graphs_data,
            ordering=ordering,
            color_encoding=color_var.get(),
            colormap='turbo',
            mapper_type='linear',
            labels=label_var.get(),
            pixel_size=40,
            ax=ax
        )

        for widget in rug_canvas.winfo_children(): widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=rug_canvas)
        canvas.draw(); canvas.get_tk_widget().pack(fill='both', expand=True)

        enable_hover(ax, fig, graphs_data, ordering, color_mapping)
        enable_click(ax, fig, graphs_data, ordering, color_mapping)
        return fig

    ttk.Button(control_frame, text="Render Rug", command=render_rug).grid(row=1,column=0,columnspan=2,pady=10)

# ---------------- GUI Entry Point ----------------
def start_gui():
    global graphs_data

    root = tk.Tk()
    root.title("NetworkRug Real Data Viewer")
    root.geometry("400x200")

    ttk.Label(root, text="Load real network JSON:").pack(pady=10)
    filename_var = tk.StringVar(value="contact_pattern_2012.json")
    ttk.Entry(root, textvariable=filename_var, width=40).pack(pady=5)

    def load_and_open():
        nonlocal graphs_data
        try:
            # load_real_network returns the dict of graphs
            graphs_data = load_real_network(filename_var.get())
        except Exception as e:
            print("Error loading network:", e)
            return
        open_rug_window()  # open the NetworkRug viewer

    ttk.Button(root, text="Load & Open NetworkRug", command=load_and_open).pack(pady=20)

    root.mainloop()
