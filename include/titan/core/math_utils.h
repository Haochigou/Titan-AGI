#pragma once
#include "types.h"
#include <algorithm>
#include <cmath>

namespace titan::core {

class StateInterpolator {
public:
    static double getAlpha(TimePoint t1, TimePoint t2, TimePoint t_query) {
        auto total = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto part = std::chrono::duration_cast<std::chrono::microseconds>(t_query - t1).count();
        if (total <= 0) return 0.0;
        return std::max(0.0, std::min(1.0, (double)part / total));
    }

    static RobotState interpolate(const RobotState& s1, const RobotState& s2, TimePoint t) {
        RobotState res;
        res.timestamp = t;
        double alpha = getAlpha(s1.timestamp, s2.timestamp, t);

        res.joint_pos = s1.joint_pos + (s2.joint_pos - s1.joint_pos) * alpha;
        res.joint_vel = s1.joint_vel + (s2.joint_vel - s1.joint_vel) * alpha;
        res.ee_pos    = s1.ee_pos    + (s2.ee_pos - s1.ee_pos) * alpha;
        res.ee_rot = s1.ee_rot.slerp(alpha, s2.ee_rot); // SLERP

        return res;
    }

    static RobotState extrapolate(const RobotState& last, double dt_sec) {
        RobotState res = last;
        res.timestamp = last.timestamp + std::chrono::microseconds((long)(dt_sec * 1e6));
        res.joint_pos += last.joint_vel * dt_sec;
        // 旋转外推更复杂，这里简化
        return res;
    }
};

} // namespace titan::core