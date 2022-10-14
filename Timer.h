#ifndef SOKOBAN3D_TIMER_H
#define SOKOBAN3D_TIMER_H

#include <chrono>

using std::chrono::time_point;
using std::chrono::duration;

class Timer {
    time_point<std::chrono::steady_clock> starttime;
    duration<float> elapsed = duration<float>(0);
    bool running = false;

public:
    float get();
    void start();
    void stop();
    void reset();
    float round();
};


#endif //SOKOBAN3D_TIMER_H
