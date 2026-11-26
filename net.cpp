#include "net.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <cstring>
#include <cstdint>

static SOCKET gSock = INVALID_SOCKET;
static PROCESS_INFORMATION gProc{};
static HANDLE gJob = nullptr;

static std::string exeDir() {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf, n);
    size_t s = p.find_last_of("\\/");
    return s == std::string::npos ? std::string(".") : p.substr(0, s);
}

static float readF32(const unsigned char* p) {
    float v; std::memcpy(&v, p, 4); return v;
}
static uint32_t readU32(const unsigned char* p) {
    uint32_t v; std::memcpy(&v, p, 4); return v;
}
static uint16_t readU16(const unsigned char* p) {
    uint16_t v; std::memcpy(&v, p, 2); return v;
}

bool netInit(int port) {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
    gSock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (gSock == INVALID_SOCKET) return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((u_short)port);
    if (bind(gSock, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        closesocket(gSock); gSock = INVALID_SOCKET; return false;
    }
    u_long nb = 1;
    ioctlsocket(gSock, FIONBIO, &nb);
    int rcv = 4 * 1024 * 1024;
    setsockopt(gSock, SOL_SOCKET, SO_RCVBUF, (const char*)&rcv, sizeof(rcv));
    return true;
}

bool netPoll(TrackerFrame& f) {
    f.handPackets = 0;
    f.imgUpdated = false;
    if (gSock == INVALID_SOCKET) return false;
    unsigned char buf[66000];
    for (;;) {
        int n = recv(gSock, (char*)buf, sizeof(buf), 0);
        if (n <= 0) break;
        if (n >= 13 && std::memcmp(buf, "HND2", 4) == 0) {
            int nh = buf[12];
            size_t need = 13 + (size_t)nh * (5 + 126 * 4);
            if (nh < 0 || nh > 2 || n < (int)need) continue;
            f.handFrame = readU32(buf + 4);
            f.camAspect = readF32(buf + 8);
            f.numHands = nh;
            const unsigned char* p = buf + 13;
            for (int i = 0; i < nh; i++) {
                f.hands[i].handedness = p[0];
                f.hands[i].score = readF32(p + 1);
                p += 5;
                for (int j = 0; j < 21; j++)
                    for (int k = 0; k < 3; k++) { f.hands[i].lm[j][k] = readF32(p); p += 4; }
                for (int j = 0; j < 21; j++)
                    for (int k = 0; k < 3; k++) { f.hands[i].wlm[j][k] = readF32(p); p += 4; }
            }
            f.handPackets++;
        } else if (n >= 16 && std::memcmp(buf, "IMG1", 4) == 0) {
            uint16_t w = readU16(buf + 8), h = readU16(buf + 10);
            uint16_t rs = readU16(buf + 12), rc = readU16(buf + 14);
            size_t need = 16 + (size_t)rc * w * 3;
            if (w == 0 || h == 0 || rs + rc > h || n < (int)need) continue;
            if (f.imgW != w || f.imgH != h) { f.imgW = w; f.imgH = h; f.img.assign((size_t)w * h * 3, 0); }
            std::memcpy(f.img.data() + (size_t)rs * w * 3, buf + 16, (size_t)rc * w * 3);
            f.imgUpdated = true;
        }
    }
    return f.handPackets > 0;
}

void netShutdown() {
    if (gSock != INVALID_SOCKET) { closesocket(gSock); gSock = INVALID_SOCKET; }
    WSACleanup();
}

bool trackerLaunch(const std::string& args, bool fallbackToPathPython) {
    std::string dir = exeDir();
    std::string py = dir + "\\.venv\\Scripts\\python.exe";
    if (GetFileAttributesA(py.c_str()) == INVALID_FILE_ATTRIBUTES) {
        if (!fallbackToPathPython) return false;
        py = "python";
    }
    std::string cmd = "\"" + py + "\" \"" + dir + "\\tracker.py\" " + args +
                      " --log \"" + dir + "\\tracker.log\"";
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW,
                        nullptr, dir.c_str(), &si, &gProc))
        return false;
    gJob = CreateJobObjectA(nullptr, nullptr);
    if (gJob) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(gJob, JobObjectExtendedLimitInformation, &info, sizeof(info));
        AssignProcessToJobObject(gJob, gProc.hProcess);
    }
    return true;
}

void trackerKill() {
    if (gProc.hProcess) {
        TerminateProcess(gProc.hProcess, 0);
        CloseHandle(gProc.hProcess);
        CloseHandle(gProc.hThread);
        gProc.hProcess = nullptr;
    }
    if (gJob) { CloseHandle(gJob); gJob = nullptr; }
}
