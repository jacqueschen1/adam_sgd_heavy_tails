"""Helpers and color definitions"""
import numpy as np


def rgb_to_unit(xs):
    """
    Convert a list of RGB numbers [1, 255] to a list of unit [0, 1]
    """
    return [x / 255.0 for x in xs]


COLORS = {
    "PT": {
        "HC": {
            "white": rgb_to_unit([255, 255, 255]),
            "yellow": rgb_to_unit([221, 170, 51]),
            "red": rgb_to_unit([187, 85, 102]),
            "blue": rgb_to_unit([0, 68, 136]),
            "black": rgb_to_unit([0, 0, 0]),
        },
        "VB": {
            "blue": rgb_to_unit([0, 119, 187]),
            "red": rgb_to_unit([204, 51, 17]),
            "orange": rgb_to_unit([238, 119, 51]),
            "cyan": rgb_to_unit([51, 187, 238]),
            "teal": rgb_to_unit([0, 153, 136]),
            "magenta": rgb_to_unit([238, 51, 119]),
            "grey": rgb_to_unit([187, 187, 187]),
        },
        "BR": {
            "blue": rgb_to_unit([68, 119, 170]),
            "cyan": rgb_to_unit([102, 204, 238]),
            "green": rgb_to_unit([34, 136, 51]),
            "yellow": rgb_to_unit([204, 187, 68]),
            "red": rgb_to_unit([238, 102, 119]),
            "purple": rgb_to_unit([170, 51, 119]),
            "gray": rgb_to_unit([187, 187, 187]),
        },
    }
}
COLORS["CC"] = [
    COLORS["PT"]["VB"]["blue"],
    COLORS["PT"]["VB"]["red"],
    COLORS["PT"]["HC"]["yellow"],
    COLORS["PT"]["VB"]["cyan"],
    COLORS["PT"]["VB"]["orange"],
    COLORS["PT"]["VB"]["magenta"],
]

#
# def rgb2hsb(rgb):
#
#    min_rgb = np.min(rgb)
#    max_rgb = np.max(rgb)
#
#    v = max_rgb
#    delta = max_rgb - min_rgb
#
#    if delta <  0.00001:
#        return [0, 0, v]
#
#    s = 0
#    if max_rgb == 0.0:
#        s = delta/max_rgb
#    else:
#        return [np.nan, 0, v]
#
#    max_idx = np.argmax(rgb)
#    h = 0
#    if max_idx == 0:
#        h = (rgb[1] - rgb[2]) /delta
#    if max_idx == 1:
#        h = 2.0 + (rgb[2] - rgb[0]) /delta
#    if max_idx == 2:
#        h = 4.0 + (rgb[0] - rgb[1]) /delta
#    h *= 60.0
#    if h < 0.0:
#        h += 360.0
#
#    return [h, s, v]
# }
#
#
# rgb hsv2rgb(hsv in)
# {
#    double      hh, p, q, t, ff;
#    long        i;
#    rgb         out;
#
#    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
#        out.r = in.v;
#        out.g = in.v;
#        out.b = in.v;
#        return out;
#    }
#    hh = in.h;
#    if(hh >= 360.0) hh = 0.0;
#    hh /= 60.0;
#    i = (long)hh;
#    ff = hh - i;
#    p = in.v * (1.0 - in.s);
#    q = in.v * (1.0 - (in.s * ff));
#    t = in.v * (1.0 - (in.s * (1.0 - ff)));
#
#    switch(i) {
#    case 0:
#        out.r = in.v;
#        out.g = t;
#        out.b = p;
#        break;
#    case 1:
#        out.r = q;
#        out.g = in.v;
#        out.b = p;
#        break;
#    case 2:
#        out.r = p;
#        out.g = in.v;
#        out.b = t;
#        break;
#
#    case 3:
#        out.r = p;
#        out.g = q;
#        out.b = in.v;
#        break;
#    case 4:
#        out.r = t;
#        out.g = p;
#        out.b = in.v;
#        break;
#    case 5:
#    default:
#        out.r = in.v;
#        out.g = p;
#        out.b = q;
#        break;
#    }
#    return out;
# }
