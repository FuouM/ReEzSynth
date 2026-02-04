#pragma once

#include "../index_vector.hpp"
#include <vector>
#include <algorithm>

namespace ebsynth
{

    struct PatchCoord
    {
        int x, y;
        float last_error = 1e30f;
        int stable_iters = 0;
    };

    // Accumulator for incremental voting
    struct VoteAccumulator
    {
        float sumColor[4] = {0, 0, 0, 0};
        float sumWeight = 0;
    };

    class PatchTracker
    {
    public:
        PatchTracker(int width, int height)
            : m_width(width), m_height(height)
        {
            reset(width, height);
        }

        void reset(int width, int height)
        {
            m_width = width;
            m_height = height;
            m_active_patches.clear();
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    m_active_patches.push_back({x, y, 1e30f, 0});
                }
            }
        }

        // Prune patches using std::remove_if to maintain scanline order
        void prune(int max_stable_iters)
        {
            auto it = std::remove_if(m_active_patches.begin(), m_active_patches.end(),
                                     [max_stable_iters](const PatchCoord &p)
                                     {
                                         return p.stable_iters >= max_stable_iters;
                                     });
            m_active_patches.erase(it, m_active_patches.end());
        }

        std::vector<PatchCoord> &getActivePatches()
        {
            return m_active_patches;
        }

        size_t size() const
        {
            return m_active_patches.size();
        }

    private:
        int m_width, m_height;
        std::vector<PatchCoord> m_active_patches; // Using std::vector to maintain order
    };

} // namespace ebsynth
