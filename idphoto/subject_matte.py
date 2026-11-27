"""Conservative ownership of a whole-image matte by detected faces."""

import cv2
import numpy as np

from .reacquiring_landmarks import _extent


def select_subject_alpha(alpha, faces, selected_index):
    """Return (selected alpha, status); refuse shared or unexplained foreground."""
    if not faces or selected_index is None:
        return None, "no_face"
    if not 0 <= selected_index < len(faces):
        raise ValueError("selected face index is invalid")
    if not np.any(alpha <= .02):
        return None, "ambiguous"
    height, width = alpha.shape
    foreground = (alpha > .02).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    face_labels = []
    for face in faces:
        x, y, _ = _extent(face)
        x, y = round(x), round(y)
        if not (0 <= x < width and 0 <= y < height):
            return None, "ambiguous"
        component = int(labels[y, x])
        if not component:
            return None, "ambiguous"
        face_labels.append(component)
    if len(set(face_labels)) != len(face_labels):
        return None, "ambiguous"
    minimum_area = max(64, round(alpha.size * .0005))
    for index in range(1, count):
        if index not in face_labels and stats[index, cv2.CC_STAT_AREA] >= minimum_area:
            return None, "ambiguous"
    if len(face_labels) == 1:
        return alpha, "ready"

    _, nearest = cv2.distanceTransformWithLabels(
        1 - foreground, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    x, y, _ = _extent(faces[selected_index])
    target_label = int(nearest[round(y), round(x)])
    if target_label == 0:
        return None, "ambiguous"
    selected = np.where(nearest == target_label, alpha, 0).astype(np.float32)
    return selected, "ready"
