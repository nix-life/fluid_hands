#pragma once
#include <string>
#include <vector>

struct HandData { int handedness; float score; float lm[21][3]; float wlm[21][3]; };
struct TrackerFrame {
    int numHands = 0; HandData hands[2]; float camAspect = 4.0f / 3.0f; unsigned handFrame = 0;
    int imgW = 0, imgH = 0; std::vector<unsigned char> img; bool imgUpdated = false;
    int handPackets = 0;
};
bool netInit(int port);
bool netPoll(TrackerFrame& f);
void netShutdown();
bool trackerLaunch(const std::string& args, bool fallbackToPathPython);
void trackerKill();
