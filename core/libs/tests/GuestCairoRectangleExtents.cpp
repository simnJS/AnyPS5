#include "prx/libcairo/include/Path.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string_view>
#include <vector>

namespace {

struct Point {
    double x;
    double y;
};

struct Extents {
    double x1;
    double y1;
    double x2;
    double y2;
};

void Same(double actual, double expected, double tolerance) {
    if (tolerance == 0) assert(actual == expected && std::signbit(actual) == std::signbit(expected));
    else assert(std::abs(actual - expected) <= tolerance);
}

void Expect(CairoContext* context, std::string_view kinds, const std::vector<Point>& points,
            const Extents& extents, double tolerance = 0) {
    CairoPath* path = cairo_copy_path(context);
    assert(path && path->status == 0);
    std::size_t kind = 0;
    std::size_t point = 0;
    for (int i = 0; i < path->num_data; i += path->data[i].header.length) {
        assert(kind < kinds.size());
        const int type = std::string_view("MLCZ").find(kinds[kind++]);
        assert(path->data[i].header.type == type);
        for (int j = 1; j < path->data[i].header.length; ++j) {
            assert(point < points.size());
            Same(path->data[i + j].point.x, points[point].x, tolerance);
            Same(path->data[i + j].point.y, points[point].y, tolerance);
            ++point;
        }
    }
    assert(kind == kinds.size() && point == points.size());
    cairo_path_destroy(path);
    Extents actual{-1, -1, -1, -1};
    cairo_path_extents(context, &actual.x1, &actual.y1, &actual.x2, &actual.y2);
    Same(actual.x1, extents.x1, tolerance);
    Same(actual.y1, extents.y1, tolerance);
    Same(actual.x2, extents.x2, tolerance);
    Same(actual.y2, extents.y2, tolerance);
    cairo_new_path(context);
}

CairoContext* Create(double scaleX = 1, double scaleY = 1) {
    CairoSurface* surface = cairo_image_surface_create(0, 8, 8);
    cairo_surface_set_device_scale(surface, scaleX, scaleY);
    CairoContext* context = cairo_create(surface);
    cairo_surface_destroy(surface);
    return context;
}

void CheckRectangles() {
    CairoContext* context = Create();
    cairo_rectangle(context, 1.1, 2.2, 3.3, 4.4);
    Expect(context, "MLLLZM",
           {{0x1.1ap+0, 0x1.198p+1}, {0x1.19cp+2, 0x1.198p+1}, {0x1.19cp+2, 0x1.a64p+2},
            {0x1.1ap+0, 0x1.a64p+2}, {0x1.1ap+0, 0x1.198p+1}},
           {0x1.1ap+0, 0x1.198p+1, 0x1.19cp+2, 0x1.a64p+2});

    cairo_rectangle(context, 5, 6, 7, 0);
    Expect(context, "MLZM", {{5, 6}, {12, 6}, {5, 6}}, {5, 6, 12, 6});
    cairo_rectangle(context, 5, 6, 7, 1.0 / 1024);
    Expect(context, "MLZM", {{5, 6}, {12, 6}, {5, 6}}, {5, 6, 12, 6});
    cairo_rectangle(context, 5, 6, 7, 1.0 / 256);
    Expect(context, "MLLLZM", {{5, 6}, {12, 6}, {12, 6 + 1.0 / 256}, {5, 6 + 1.0 / 256}, {5, 6}},
           {5, 6, 12, 6 + 1.0 / 256});
    cairo_rectangle(context, 10, 10, -4, -3);
    Expect(context, "MLLLZM", {{10, 10}, {6, 10}, {6, 7}, {10, 7}, {10, 10}}, {6, 7, 10, 10});
    cairo_rectangle(context, 3, 4, 0, 0);
    Expect(context, "MZM", {{3, 4}, {3, 4}}, {3, 4, 3, 4});
    cairo_rectangle(context, 3, 4, 0, 5);
    Expect(context, "MLZM", {{3, 4}, {3, 9}, {3, 4}}, {3, 4, 3, 9});

    const double infinity = std::numeric_limits<double>::infinity();
    cairo_rectangle(context, 0, 0, 1e7, 10);
    Expect(context, "MLLLZM", {{0, 0}, {-0x1.9da6p+22, 0}, {-0x1.9da6p+22, 10}, {0, 10}, {0, 0}},
           {-0x1.9da6p+22, 0, 0, 10});
    cairo_rectangle(context, 8388000, 0, 1000, 10);
    Expect(context, "MLLLZM",
           {{0x1.fff68p+22, 0}, {-0x1.fff9ep+22, 0}, {-0x1.fff9ep+22, 10}, {0x1.fff68p+22, 10}, {0x1.fff68p+22, 0}},
           {-0x1.fff9ep+22, 0, 0x1.fff68p+22, 10});
    cairo_rectangle(context, 1, 2, infinity, 10);
    Expect(context, "MLZM", {{1, 2}, {1, 12}, {1, 2}}, {1, 2, 1, 12});
    cairo_rectangle(context, 1, 2, 3, -infinity);
    Expect(context, "MLZM", {{1, 2}, {4, 2}, {1, 2}}, {1, 2, 4, 2});
    cairo_rectangle(context, 5, 6, 0.3, 0.7);
    Expect(context, "MLLLZM",
           {{5, 6}, {0x1.534p+2, 6}, {0x1.534p+2, 0x1.accp+2}, {5, 0x1.accp+2}, {5, 6}},
           {5, 6, 0x1.534p+2, 0x1.accp+2});
    cairo_rectangle(context, 5, 6, 1.0 / 512, 1.0 / 512);
    Expect(context, "MZM", {{5, 6}, {5, 6}}, {5, 6, 5, 6});
    cairo_rectangle(context, 5, 6, 3.0 / 512, 3.0 / 512);
    Expect(context, "MLLLZM",
           {{5, 6}, {0x1.408p+2, 6}, {0x1.408p+2, 0x1.808p+2}, {5, 0x1.808p+2}, {5, 6}},
           {5, 6, 0x1.408p+2, 0x1.808p+2});
    cairo_rectangle(context, -0.001, -0.001, 1, 1);
    Expect(context, "MLLLZM", {{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}}, {0, 0, 1, 1});
    cairo_move_to(context, -0.001, 3);
    cairo_line_to(context, 2, -0.001);
    Expect(context, "ML", {{0, 3}, {2, 0}}, {0, 0, 2, 3});

    cairo_move_to(context, 1, 1);
    cairo_line_to(context, 2, 2);
    cairo_rectangle(context, 3, 4, 5, 6);
    cairo_line_to(context, 0, 0);
    Expect(context, "MLMLLLZML",
           {{1, 1}, {2, 2}, {3, 4}, {8, 4}, {8, 10}, {3, 10}, {3, 4}, {0, 0}}, {0, 0, 8, 10});

    cairo_save(context);
    cairo_translate(context, 7, -3);
    cairo_scale(context, 2, 3);
    cairo_rectangle(context, 1.25, 1.5, 2, 4);
    Expect(context, "MLLLZM", {{1.25, 1.5}, {3.25, 1.5}, {3.25, 5.5}, {1.25, 5.5}, {1.25, 1.5}},
           {1.25, 1.5, 3.25, 5.5});
    cairo_restore(context);

    cairo_save(context);
    cairo_translate(context, 4, 5);
    cairo_rotate(context, 0.3);
    cairo_rectangle(context, 1, 2, 3, 4);
    Expect(context, "MLLLZM",
           {{0x1.ffa1633cfa5p-1, 0x1.00240ff66596ap+1}, {0x1.000760c02198fp+2, 0x1.001d9adc61a96p+1},
            {0x1.ffd56313bb366p+1, 0x1.8005f3aba03d7p+2}, {0x1.febbe98ada628p-1, 0x1.80092e38a2341p+2},
            {0x1.ffa1633cfa5p-1, 0x1.00240ff66596ap+1}},
           {-0x1.0d36c69501e2p-3, 0x1.275eb21f95a5cp+0, 0x1.4848942624f43p+2, 0x1.b6404f1eed9f5p+2}, 1e-12);
    cairo_restore(context);
    cairo_destroy(context);

    context = Create(2, 0.5);
    cairo_rectangle(context, 1.1, 2.2, 3.3, 4.4);
    Expect(context, "MLLLZM",
           {{0x1.198p+0, 0x1.1ap+1}, {0x1.19ap+2, 0x1.1ap+1}, {0x1.19ap+2, 0x1.a68p+2},
            {0x1.198p+0, 0x1.a68p+2}, {0x1.198p+0, 0x1.1ap+1}},
           {0x1.198p+0, 0x1.1ap+1, 0x1.19ap+2, 0x1.a68p+2});
    cairo_move_to(context, 0.3, 0.7);
    cairo_curve_to(context, 10, -20, 20, 30, 30, 0.1);
    Expect(context, "MC", {{0x1.34p-2, 0x1.68p-1}, {10, -20}, {20, 30}, {30, 0x1.ap-4}},
           {0x1.34p-2, -0x1.1fp+2, 30, 0x1.3dp+3});
    cairo_destroy(context);
}

void CheckClosePath() {
    CairoContext* context = Create();
    cairo_move_to(context, 0, 0);
    cairo_line_to(context, 10, 0);
    cairo_line_to(context, 0, 0);
    cairo_close_path(context);
    Expect(context, "MLZM", {{0, 0}, {10, 0}, {0, 0}}, {0, 0, 10, 0});
    cairo_move_to(context, 0, 0);
    cairo_line_to(context, 10, 0);
    cairo_line_to(context, 5, 0);
    cairo_close_path(context);
    Expect(context, "MLZM", {{0, 0}, {10, 0}, {0, 0}}, {0, 0, 10, 0});
    cairo_move_to(context, 0, 0);
    cairo_line_to(context, 10, 0);
    cairo_line_to(context, 10, 5);
    cairo_close_path(context);
    cairo_line_to(context, 3, 3);
    Expect(context, "MLLZML", {{0, 0}, {10, 0}, {10, 5}, {0, 0}, {3, 3}}, {0, 0, 10, 5});
    cairo_move_to(context, 1, 1);
    cairo_line_to(context, 1, 1);
    cairo_close_path(context);
    Expect(context, "MZM", {{1, 1}, {1, 1}}, {1, 1, 1, 1});
    cairo_move_to(context, 3, 4);
    cairo_close_path(context);
    Expect(context, "MZM", {{3, 4}, {3, 4}}, {3, 4, 3, 4});
    cairo_move_to(context, 0, 0);
    cairo_curve_to(context, 5, 5, 5, -5, 0, 0);
    cairo_close_path(context);
    Expect(context, "MCZM", {{0, 0}, {5, 5}, {5, -5}, {0, 0}, {0, 0}}, {0, -0x1.72p+0, 0x1.ep+1, 0x1.72p+0});
    cairo_destroy(context);
}

void CheckExtents() {
    CairoContext* context = Create();
    Expect(context, "", {}, {0, 0, 0, 0});
    cairo_move_to(context, 3, 4);
    Expect(context, "M", {{3, 4}}, {0, 0, 0, 0});
    cairo_move_to(context, 3, 4);
    cairo_move_to(context, -2, 9);
    cairo_line_to(context, 1, 1);
    cairo_move_to(context, 50, 50);
    Expect(context, "MLM", {{-2, 9}, {1, 1}, {50, 50}}, {-2, 1, 1, 9});
    cairo_new_sub_path(context);
    cairo_move_to(context, 2, 2);
    cairo_line_to(context, 3, 3);
    cairo_new_sub_path(context);
    cairo_move_to(context, -9, -9);
    Expect(context, "MLM", {{2, 2}, {3, 3}, {-9, -9}}, {2, 2, 3, 3});

    cairo_move_to(context, 0, 0);
    cairo_curve_to(context, 10, -20, 20, 30, 30, 0);
    Expect(context, "MC", {{0, 0}, {10, -20}, {20, 30}, {30, 0}}, {0, -0x1.37cp+2, 30, 0x1.3b4p+3});
    cairo_move_to(context, 0, 0);
    cairo_line_to(context, 40, 40);
    cairo_curve_to(context, 10, 10, 20, 30, 30, 0);
    Expect(context, "MLC", {{0, 0}, {40, 40}, {10, 10}, {20, 30}, {30, 0}}, {0, 0, 40, 40});

    cairo_save(context);
    cairo_rotate(context, 0.5);
    cairo_move_to(context, 1, 2);
    cairo_curve_to(context, 3, -7, 9, 11, 5, 1);
    Expect(context, "MC",
           {{0x1.ff9ab69664f5ap-1, 0x1.0005c7dbe5fefp+1}, {0x1.800d883265c5cp+1, -0x1.bfe4667c31d55p+2},
            {0x1.2005289cec266p+3, 0x1.6001f6002b86ap+3}, {0x1.400dffee650f6p+2, 0x1.ff04b8d72cf18p-1}},
           {0x1.e568683d2b08ap-3, -0x1.4f1a955a1daacp+0, 0x1.a7f85d303f248p+2, 0x1.7896283638d2bp+2}, 1e-12);
    cairo_restore(context);

    cairo_rectangle(context, 1, 2, 3, 4);
    double x1 = -1, y2 = -1;
    cairo_path_extents(context, &x1, nullptr, nullptr, &y2);
    assert(x1 == 1 && y2 == 6);
    cairo_path_extents(context, nullptr, nullptr, nullptr, nullptr);
    cairo_restore(context);
    double values[4] = {-1, -1, -1, -1};
    cairo_path_extents(context, &values[0], &values[1], &values[2], &values[3]);
    assert(values[0] == 0 && values[1] == 0 && values[2] == 0 && values[3] == 0);
    cairo_rectangle(context, 1, 2, 3, 4);
    CairoPath* path = cairo_copy_path(context);
    assert(path && path->status == 2 && path->num_data == 0);
    cairo_path_destroy(path);
    cairo_destroy(context);

    double nullValues[4] = {-1, -1, -1, -1};
    cairo_path_extents(nullptr, &nullValues[0], &nullValues[1], &nullValues[2], &nullValues[3]);
    assert(nullValues[0] == 0 && nullValues[1] == 0 && nullValues[2] == 0 && nullValues[3] == 0);
    cairo_rectangle(nullptr, 1, 2, 3, 4);
}

void CheckOperator() {
    CairoContext* context = Create();
    assert(cairo_get_operator(context) == 2);
    cairo_set_operator(context, 13);
    assert(cairo_get_operator(context) == 13);
    cairo_save(context);
    cairo_set_operator(context, 0);
    assert(cairo_get_operator(context) == 0);
    cairo_set_operator(context, -5);
    assert(cairo_get_operator(context) == -5);
    cairo_set_operator(context, 99);
    assert(cairo_get_operator(context) == 99);
    cairo_restore(context);
    assert(cairo_get_operator(context) == 13);
    cairo_restore(context);
    assert(cairo_get_operator(context) == 2);
    cairo_set_operator(context, 7);
    assert(cairo_get_operator(context) == 2);
    cairo_destroy(context);

    context = cairo_create(nullptr);
    assert(cairo_get_operator(context) == 2);
    cairo_set_operator(context, 4);
    assert(cairo_get_operator(context) == 2);
    cairo_destroy(context);
    assert(cairo_get_operator(nullptr) == 2);
    cairo_set_operator(nullptr, 4);
}

}

int main() {
    CheckRectangles();
    CheckClosePath();
    CheckExtents();
    CheckOperator();
    std::puts("cairo rectangle, operator and path extents tests passed");
    return 0;
}
