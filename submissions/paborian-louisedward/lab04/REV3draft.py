import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AutoUpdatingSolver3D:

    def __init__(
        self,
        solver_excel,
        master_excel,
        unit_system,
        material_type,
        material_grade,
        member_size,
    ):
        """Single source of truth constructor.

        No default values are hardcoded here.
        """
        self.solver_excel = solver_excel
        self.master_excel = master_excel
        self.unit_system = unit_system.strip().title()
        self.material_type = material_type.strip()
        self.material_grade = material_grade.strip()
        self.member_size = member_size.strip()
        self.props = {}

    def load_excel_data(self):
        """Extracts node geometry, AISC dimensions, and material values directly from Excel sources."""
        if not os.path.exists(self.solver_excel):
            raise FileNotFoundError(
                f"Solver file not found at: {self.solver_excel}"
            )
        if not os.path.exists(self.master_excel):
            raise FileNotFoundError(
                f"Master library file not found at: {self.master_excel}"
            )

        is_metric = self.unit_system.lower() == "metric"

        # 1. READ GEOMETRY & DOF MAPS FROM SOLVER WORKBOOK
        xls_sol = pd.ExcelFile(self.solver_excel)
        df_nodes = pd.read_excel(xls_sol, sheet_name="Nodes")
        df_nodes.columns = df_nodes.iloc[2]
        df_nodes = df_nodes.iloc[3:].dropna(how="all").reset_index(drop=True)

        df_members = pd.read_excel(xls_sol, sheet_name="Member Incidences")
        df_members.columns = df_members.iloc[2]
        df_members = (
            df_members.iloc[3:].dropna(how="all").reset_index(drop=True)
        )

        # 2. READ AISC SECTION DATABASE
        xls_mas = pd.ExcelFile(self.master_excel)
        aisc_sheet = [
            s for s in xls_mas.sheet_names if "aisc" in s and "Dat" in s
        ][0]
        df_aisc = pd.read_excel(xls_mas, sheet_name=aisc_sheet)

        section_match = df_aisc[
            (df_aisc["AISC_Manual_Label"] == self.member_size)
            | (df_aisc["EDI_Std_Nomenclature"] == self.member_size)
        ]

        if section_match.empty:
            raise ValueError(
                f"Member size '{self.member_size}' was not found in the AISC section database."
            )

        if is_metric:
            A = float(section_match["A.1"].values[0]) * 1e-6  # m²
            Ix = float(section_match["Ix.1"].values[0]) * 1e-8  # m⁴
            Iy = float(section_match["Iy.1"].values[0]) * 1e-8  # m⁴
            J = float(section_match["J.1"].values[0]) * 1e-8  # m⁴
        else:
            A = float(section_match["A"].values[0])  # in²
            Ix = float(section_match["Ix"].values[0])  # in⁴
            Iy = float(section_match["Iy"].values[0])  # in⁴
            J = float(section_match["J"].values[0])  # in⁴

        # 3. READ MATERIAL PROPERTIES LIBRARY
        df_mat_raw = pd.read_excel(xls_mas, sheet_name="RISA_All Materials")
        header_idx = 2
        for idx, row in df_mat_raw.iterrows():
            if "Label" in row.values:
                header_idx = idx
                break

        df_mat = pd.read_excel(
            xls_mas, sheet_name="RISA_All Materials", skiprows=header_idx + 1
        )

        mat_match = df_mat[
            (
                df_mat["Label"].astype(str).str.strip().str.lower()
                == self.material_grade.lower()
            )
            & (
                df_mat["Category"].astype(str).str.strip().str.lower()
                == self.material_type.lower()
            )
        ]

        if mat_match.empty:
            mat_match = df_mat[
                df_mat["Label"].astype(str).str.strip().str.lower()
                == self.material_grade.lower()
            ]

        if mat_match.empty:
            raise ValueError(
                f"Material grade '{self.material_grade}' was not found in the material library."
            )

        if is_metric:
            E = float(mat_match["E [MPa]"].values[0]) * 1e6  # Pa
            G = float(mat_match["G [MPa]"].values[0]) * 1e6  # Pa
            density = float(
                mat_match["Mass Density [kg/m³]"].values[0]
            )  # kg/m³
            yield_stress = (
                float(mat_match["Yield / f'c / f'm [MPa]"].values[0]) * 1e6
            )  # Pa
        else:
            E = float(mat_match["E [ksi]"].values[0])  # ksi
            G = float(mat_match["G [ksi]"].values[0])  # ksi
            density = float(
                mat_match["Density [k/ft³]"].values[0]
            )  # k/ft³
            yield_stress = float(
                mat_match["Yield / f'c / f'm [ksi]"].values[0]
            )  # ksi

        self.props = {
            "unit_system": "Metric" if is_metric else "Imperial",
            "material_type": self.material_type,
            "material_grade": self.material_grade,
            "member_size": self.member_size,
            "A": A,
            "Ix": Ix,
            "Iy": Iy,
            "J": J,
            "E": E,
            "G": G,
            "density": density,
            "yield_stress": yield_stress,
            "nodes": df_nodes,
            "members": df_members,
        }
        return self.props

    def plot_structure(self):
        """Generates 3D visualization using standard Cartesian orientation (Y = vertical height, Z = depth)."""
        if not self.props:
            self.load_excel_data()

        fig = plt.figure(figsize=(12, 7.5), dpi=120)
        ax = fig.add_subplot(111, projection="3d")

        bg_color = "#f8f9fa"
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#e2e8f0")
        ax.yaxis.pane.set_edgecolor("#e2e8f0")
        ax.zaxis.pane.set_edgecolor("#e2e8f0")

        ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")

        nodes_df = self.props["nodes"]
        members_df = self.props["members"]

        node_coords = {}

        # 1. Map to standard Cartesian axes (X: horizontal, Y: vertical elevation, Z: depth)
        for _, row in nodes_df.iterrows():
            nid = int(row.iloc[0])
            raw_x, raw_y, raw_z = (
                float(row.iloc[1]),
                float(row.iloc[2]),
                float(row.iloc[3]),
            )

            # Standard Cartesian Mapping:
            plot_x = raw_x
            plot_y = raw_z  # Depth axis
            plot_z = raw_y  # Vertical height axis

            node_coords[nid] = (plot_x, plot_y, plot_z)

            ax.scatter(
                plot_x,
                plot_y,
                plot_z,
                color="#1e3a8a",
                s=60,
                edgecolors="#3b82f6",
                linewidth=1.2,
                zorder=5,
            )

            # Dynamic label offsets along the vertical elevation axis
            z_off, x_off = (0.35, -0.25) if plot_z > 0 else (-0.45, 0.25)
            ax.text(
                plot_x + x_off,
                plot_y,
                plot_z + z_off,
                f"N{nid}",
                color="#0f172a",
                fontsize=9.5,
                fontweight="bold",
                zorder=6,
            )

        # 2. Plot Structural Members
        for _, row in members_df.iterrows():
            ni, nj = int(row.iloc[1]), int(row.iloc[2])
            if ni in node_coords and nj in node_coords:
                x_pts = [node_coords[ni][0], node_coords[nj][0]]
                y_pts = [node_coords[ni][1], node_coords[nj][1]]
                z_pts = [node_coords[ni][2], node_coords[nj][2]]
                ax.plot(
                    x_pts,
                    y_pts,
                    z_pts,
                    color="#dc2626",
                    linewidth=2.5,
                    alpha=0.9,
                    zorder=4,
                )

        is_metric = self.props["unit_system"] == "Metric"
        u_len = "m" if is_metric else "in"
        u_mod = "Pa" if is_metric else "ksi"
        u_dens = "kg/m³" if is_metric else "k/ft³"

        # Set z-limits based on vertical elevation
        max_height = nodes_df.iloc[:, 2].astype(float).max()
        min_height = nodes_df.iloc[:, 2].astype(float).min()
        ax.set_zlim(min_height - 1.0, max_height + 3.5)

        # Labels updated for standard Cartesian convention
        ax.set_xlabel(
            f"X Axis ({u_len})",
            labelpad=8,
            fontsize=9.5,
            fontweight="semibold",
            color="#334155",
        )
        ax.set_ylabel(
            f"Z Axis ({u_len})",
            labelpad=8,
            fontsize=9.5,
            fontweight="semibold",
            color="#334155",
        )
        ax.set_zlabel(
            f"Y Axis ({u_len})",
            labelpad=8,
            fontsize=9.5,
            fontweight="semibold",
            color="#334155",
        )

        ax.set_title(
            "3D STRUCTURAL FRAME MODEL",
            fontsize=14,
            fontweight="bold",
            color="#0f172a",
            pad=15,
        )

        # Left Callout: General Model Specs
        info_annotation = (
            f"MODEL SPECIFICATIONS\n"
            f"• Section Size  : {self.props['member_size']}\n"
            f"• Material Grade: {self.props['material_grade']} ({self.props['material_type']})\n"
            f"• Unit System   : {self.props['unit_system']} ({u_len})"
        )

        ax.text2D(
            0.02,
            0.98,
            info_annotation,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="white",
                edgecolor="#cbd5e1",
                alpha=0.95,
            ),
        )

        # Right Callout: Resolved Properties
        mat_annotation = (
            f"RESOLVED PROPERTIES\n"
            f"• Area (A) : {self.props['A']:.3e} {u_len}²\n"
            f"• Ix       : {self.props['Ix']:.3e} {u_len}⁴\n"
            f"• Iy       : {self.props['Iy']:.3e} {u_len}⁴\n"
            f"• Modulus E: {self.props['E']:.3e} {u_mod}\n"
            f"• Modulus G: {self.props['G']:.3e} {u_mod}\n"
            f"• Density  : {self.props['density']:.1f} {u_dens}\n"
            f"• Yield Fy : {self.props['yield_stress']:.3e} {u_mod}"
        )

        ax.text2D(
            0.98,
            0.98,
            mat_annotation,
            transform=ax.transAxes,
            fontsize=8.5,
            horizontalalignment="right",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="white",
                edgecolor="#cbd5e1",
                alpha=0.95,
            ),
        )

        ax.view_init(elev=20, azim=-60)
        plt.tight_layout()
        plt.show()

    def solve(self):
        """Runs the complete calculation pipeline and triggers visualization."""
        data = self.load_excel_data()

        is_metric = data["unit_system"] == "Metric"
        u_len = "m" if is_metric else "in"
        u_mod = "Pa" if is_metric else "ksi"
        u_dens = "kg/m³" if is_metric else "k/ft³"

        print("=" * 65)
        print("    3D STRUCTURAL SOLVER — DYNAMIC SINGLE SOURCE SETUP   ")
        print("=" * 65)
        print(f"  Unit System       : {data['unit_system']}")
        print(f"  Material Type     : {data['material_type']}")
        print(f"  Material Grade    : {data['material_grade']}")
        print(f"  Member Size       : {data['member_size']}")
        print("-" * 65)
        print("  RESOLVED SECTION & MATERIAL PROPERTIES:")
        print(f"   • Cross-Sectional Area (A) : {data['A']:.6e} {u_len}²")
        print(f"   • Moment of Inertia (Ix)   : {data['Ix']:.6e} {u_len}⁴")
        print(f"   • Moment of Inertia (Iy)   : {data['Iy']:.6e} {u_len}⁴")
        print(f"   • Torsional Constant (J)   : {data['J']:.6e} {u_len}⁴")
        print(f"   • Elastic Modulus (E)      : {data['E']:.3e} {u_mod}")
        print(f"   • Shear Modulus (G)        : {data['G']:.3e} {u_mod}")
        print(f"   • Mass/Unit Weight         : {data['density']:.3f} {u_dens}")
        print(
            f"   • Yield Strength (Fy)      : {data['yield_stress']:.3e} {u_mod}"
        )
        print("-" * 65)
        print(
            f"  MODEL GEOMETRY LOADED: {len(data['nodes'])} Nodes | {len(data['members'])} Members"
        )
        print("=" * 65 + "\n")

        self.plot_structure()


# ==============================================================================
# >>> YOUR ONLY CONTROL PANEL (EDIT YOUR INPUTS HERE ONLY) <<<
# ==============================================================================
if __name__ == "__main__":
    solver_path = r"C:\Users\Louis\Downloads\Structural_Solver_Rev1.xlsx"
    master_path = r"C:\Users\Louis\Downloads\Master_Reference_Library.xlsx"

    solver = AutoUpdatingSolver3D(
        solver_excel=solver_path,
        master_excel=master_path,
        unit_system="Metric",  # Options: "Metric" or "Imperial"
        material_type="Hot Rolled",  # Options: "Hot Rolled", "Cold Formed", "Concrete", etc.
        material_grade="A36 Gr.36",  # Options: "A992", "Conc3000NW", "A36 Gr.36", etc.
        member_size="W40X655",  # Options: Any AISC designation (e.g., "W14X90", "W12X26")
    )

    solver.solve()