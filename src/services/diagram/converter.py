from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.services.diagram import geometry as geo
from src.services.diagram.models.entities import GeometricPoint, Diagram

if TYPE_CHECKING:
    from src.services.diagram.optimizer import Optimizer


def build_diagram(opt: Optimizer) -> Diagram:
    """Read the optimizer's solved state and produce a ``Diagram``."""
    diagram = Diagram()

    # --- Points -----------------------------------------------------------
    for name, pt in opt.name2pt.items():
        x = pt.x.detach().cpu().item()
        y = pt.y.detach().cpu().item()
        diagram.add_point(name, GeometricPoint(x, y, name))

    # --- Triangles --------------------------------------------------------
    for key, metadata in opt.triangles_metadata.items():
        p1_name, p2_name, p3_name = key
        if all(n in diagram.points for n in (p1_name, p2_name, p3_name)):
            logger.info(f"Adding triangle {key} with equal_angles: {metadata.get('equal_angles')}")
            diagram.add_triangle(
                diagram.points[p1_name],
                diagram.points[p2_name],
                diagram.points[p3_name],
                metadata.get('equal_sides'),
                metadata.get('right_angle_at'),
                metadata.get('equal_angles'),
            )

    # --- Quadrilaterals ---------------------------------------------------
    for key, metadata in opt.quadrilaterals_metadata.items():
        p1_name, p2_name, p3_name, p4_name = key
        if all(n in diagram.points for n in (p1_name, p2_name, p3_name, p4_name)):
            diagram.add_quadrilateral(
                diagram.points[p1_name],
                diagram.points[p2_name],
                diagram.points[p3_name],
                diagram.points[p4_name],
                metadata,
            )

    # --- Circles ----------------------------------------------------------
    for center_name, info in opt.circles:
        if center_name not in diagram.points:
            continue
        center_pt = opt.name2pt[center_name]

        if info['type'] == 'incircle':
            tri_pts = info['triangle']
            p1 = opt.name2pt[tri_pts[0]]
            p2 = opt.name2pt[tri_pts[1]]
            radius = geo.dist_to_line(center_pt, p1, p2).detach().cpu().item()
            info = {**info, 'radius': radius}

        elif info['type'] == 'circumcircle':
            tri_pts = info['triangle']
            p1 = opt.name2pt[tri_pts[0]]
            radius = geo.dist(center_pt, p1).detach().cpu().item()
            info = {**info, 'radius': radius}

        diagram.add_circle(diagram.points[center_name], info)

    # --- Segments ---------------------------------------------------------
    for p1_name, p2_name in opt.segments:
        if p1_name in diagram.points and p2_name in diagram.points:
            diagram.add_segment(diagram.points[p1_name], diagram.points[p2_name])

    # --- Lines ------------------------------------------------------------
    for p1_name, p2_name in opt.lines:
        if p1_name in diagram.points and p2_name in diagram.points:
            line_name = f"line_{p1_name}_{p2_name}"
            diagram.add_line(line_name, (diagram.points[p1_name], diagram.points[p2_name]))

    # --- Angle bisectors --------------------------------------------------
    for bisector_data in getattr(opt, 'angle_bisectors_metadata', []):
        v_name = bisector_data['vertex']
        bp_name = bisector_data['bisector_point']
        if v_name in diagram.points and bp_name in diagram.points:
            diagram.angle_bisectors.append({
                'vertex': diagram.points[v_name],
                'point': diagram.points[bp_name],
                'angle_points': bisector_data.get('angle_points', []),
            })

    # --- Angle-equal assertions -------------------------------------------
    for assertion in opt.angle_equal_assertions:
        angle1 = assertion['angle1']
        angle2 = assertion['angle2']
        if all(n in diagram.points for n in angle1 + angle2):
            diagram.angle_equal_assertions.append({
                'angle1': {
                    'p1': diagram.points[angle1[0]],
                    'vertex': diagram.points[angle1[1]],
                    'p2': diagram.points[angle1[2]],
                },
                'angle2': {
                    'p1': diagram.points[angle2[0]],
                    'vertex': diagram.points[angle2[1]],
                    'p2': diagram.points[angle2[2]],
                },
            })

    # --- Angle measures ---------------------------------------------------
    for vertex_name, p1_name, p2_name, degrees in opt.angle_measures:
        if all(n in diagram.points for n in (vertex_name, p1_name, p2_name)):
            diagram.add_angle_measure(
                diagram.points[vertex_name],
                diagram.points[p1_name],
                diagram.points[p2_name],
                degrees,
            )

    return diagram
