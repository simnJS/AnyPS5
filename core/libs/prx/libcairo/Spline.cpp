/* Derived from src/cairo-spline.c@1.18.0 */
/* cairo - a vector graphics library with display and print output
 *
 * Copyright © 2002 University of Southern California
 *
 * This library is free software; you can redistribute it and/or
 * modify it either under the terms of the GNU Lesser General Public
 * License version 2.1 as published by the Free Software Foundation
 * (the "LGPL") or, at your option, under the terms of the Mozilla
 * Public License Version 1.1 (the "MPL"). If you do not alter this
 * notice, a recipient may use your version of this file under either
 * the MPL or the LGPL.
 *
 * You should have received a copy of the LGPL along with this library
 * in the file COPYING-LGPL-2.1; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Suite 500, Boston, MA 02110-1335, USA
 * You should have received a copy of the MPL along with this library
 * in the file COPYING-MPL-1.1
 *
 * The contents of this file are subject to the Mozilla Public License
 * Version 1.1 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY
 * OF ANY KIND, either express or implied. See the LGPL or the MPL for
 * the specific language governing rights and limitations.
 *
 * The Original Code is the cairo graphics library.
 *
 * The Initial Developer of the Original Code is University of Southern
 * California.
 *
 * Contributor(s):
 *	Carl D. Worth <cworth@cworth.org>
 */

#include "prx/libcairo/include/Spline.hpp"

#include <cmath>

namespace {

void FindExtremes(double a, double b, double c, double* t, int& count) {
    const auto add = [&](double candidate) {
        if (0 < candidate && candidate < 1) t[count++] = candidate;
    };
    if (a == 0) {
        if (b != 0) add(-c / (2 * b));
        return;
    }
    const double b2 = b * b;
    const double delta = b2 - a * c;
    if (delta > 0) {
        bool feasible;
        const double twoAb = 2 * a * b;
        if (twoAb >= 0)
            feasible = delta > b2 && delta < a * a + b2 + twoAb;
        else if (-b / a >= 1)
            feasible = delta < b2 && delta > a * a + b2 + twoAb;
        else
            feasible = delta < b2 || delta < a * a + b2 + twoAb;
        if (feasible) {
            const double root = std::sqrt(delta);
            add((-b - root) / a);
            add((-b + root) / a);
        }
    } else if (delta == 0) {
        add(-b / a);
    }
}

} // namespace

void CairoSplineBound(const CairoFixedPoint& p0, const CairoFixedPoint& p1,
                      const CairoFixedPoint& p2, const CairoFixedPoint& p3,
                      CairoSplinePointSink sink, void* closure) {
    double t[4];
    int count = 0;
    FindExtremes(-p0.x + 3 * p1.x - 3 * p2.x + p3.x, p0.x - 2 * p1.x + p2.x, -p0.x + p1.x, t, count);
    FindExtremes(-p0.y + 3 * p1.y - 3 * p2.y + p3.y, p0.y - 2 * p1.y + p2.y, -p0.y + p1.y, t, count);
    sink(closure, p0.x, p0.y);
    for (int i = 0; i < count; ++i) {
        const double t10 = t[i];
        const double t01 = 1 - t10;
        const double t20 = t10 * t10;
        const double t02 = t01 * t01;
        const double t30 = t20 * t10;
        const double t213 = t20 * t01 * 3;
        const double t123 = t10 * t02 * 3;
        const double t03 = t01 * t02;
        sink(closure, p0.x * t03 + p1.x * t123 + p2.x * t213 + p3.x * t30,
             p0.y * t03 + p1.y * t123 + p2.y * t213 + p3.y * t30);
    }
    sink(closure, p3.x, p3.y);
}
