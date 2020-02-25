import cv2
import six


def voc_colormap(nlabels, order='rgb'):
    colors = []
    for i in six.moves.range(nlabels):
        r, g, b = 0, 0, 0
        for j in range(8):
            if i & (1 << 0):
                r |= 1 << (7 - j)
            if i & (1 << 1):
                g |= 1 << (7 - j)
            if i & (1 << 2):
                b |= 1 << (7 - j)
            i >>= 3
        if order == 'rgb':
            colors.append([r, g, b])
        elif order == 'bgr':
            colors.append([b, g, r])
    return colors


def vis_bboxes(img, bboxes,
               labels,
               label_names=None,
               font_scale=0.8,
               thickness=1,
               font_face=cv2.FONT_HERSHEY_SIMPLEX,
               text_color=(255, 255, 255),
               max_label_num=1024):
    """Visualize bounding boxes inside image.

    """
    if len(bboxes) != len(labels):
        raise ValueError("len(bboxes) and len(labels) should be same "
                         "we get len(bboxes):{}, len(lables):{}"
                         .format(len(bboxes), len(labels)))
    colormap = voc_colormap(max_label_num)

    CV_AA = 16  # for anti-alias
    for bbox, label in zip(bboxes, labels):
        color = colormap[label % max_label_num]
        y1, x1, y2, x2 = bbox
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, CV_AA)

        img_bbox = img[y1:y2, x1:]

        if label_names is not None:
            text = label_names[label]
        else:
            text = str(label)
        size, baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness)
        cv2.rectangle(
            img_bbox, (0, 0), (size[0], size[1] + baseline),
            color=color, thickness=-1)
        cv2.putText(img_bbox, text, (0, size[1]),
                    font_face, font_scale, text_color, thickness)
    return img
