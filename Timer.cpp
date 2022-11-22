#include "Timer.h"

using std::chrono::time_point;
using std::chrono::duration;
using std::chrono::steady_clock;

float Timer::get() {
    if (this->running) {
        duration<float> d = steady_clock::now() - this->starttime;
        return (this->elapsed + d).count();
    } else {
        return this->elapsed.count();
    }
}

void Timer::reset() {
    this->elapsed = duration<float>(0);
    if (this->running) {
        this->starttime = steady_clock::now();
    }
}

void Timer::start() {
    if (this->running)
        return;
    this->running = true;
    this->starttime = steady_clock::now();
}

void Timer::stop() {
    if (!this->running)
        return;
    this->running = false;
    this->elapsed += steady_clock::now() - this->starttime;
}

float Timer::round() {
    if (!this->running)
        return 0;

    auto now = steady_clock::now();
    duration<float> d = now - this->starttime;
    elapsed += d;
    this->starttime = now;
    return d.count();
}
