#pragma once

struct CairoFixedPoint {
    double x;
    double y;
};

using CairoSplinePointSink = void (*)(void* closure, double x, double y);

void CairoSplineBound(const CairoFixedPoint& p0, const CairoFixedPoint& p1,
                      const CairoFixedPoint& p2, const CairoFixedPoint& p3,
                      CairoSplinePointSink sink, void* closure);
