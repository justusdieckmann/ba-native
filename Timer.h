#ifndef SOKOBAN3D_TIMER_H
#define SOKOBAN3D_TIMER_H

#include <chrono>

using std::chrono::time_point;
using std::chrono::duration;

class Timer {
    time_point<std::chrono::steady_clock> starttime;
    duration<double> elapsed = duration<double>(0);
    bool running = false;

public:
    double get();
    void start();
    void stop();
    void reset();
    double round();
};


#endif //SOKOBAN3D_TIMER_H
