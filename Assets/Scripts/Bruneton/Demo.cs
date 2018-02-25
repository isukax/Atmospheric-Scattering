using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Bruneton {
    public class Demo : MonoBehaviour {
        public enum Luminance {
            // Render the spectral radiance at kLambdaR, kLambdaG, kLambdaB.
            NONE,
            // Render the sRGB luminance, using an approximate (on the fly) conversion
            // from 3 spectral radiance values only (see section 14.3 in <a href=
            // "https://arxiv.org/pdf/1612.04336.pdf">A Qualitative and Quantitative
            //  Evaluation of 8 Clear Sky Models</a>).
            APPROXIMATE,
            // Render the sRGB luminance, precomputed from 15 spectral radiance values
            // (see section 4.4 in <a href=
            // "http://www.oskee.wz.cz/stranka/uploads/SCCG10ElekKmoch.pdf">Real-time
            //  Spectral Scattering in Large-scale Natural Participating Media</a>).
            PRECOMPUTED
        };

        public bool use_constant_solar_spectrum_;
        public bool use_ozone_;
        public bool use_combined_textures_;
        public bool use_half_precision_;
        public Luminance use_luminance_;
        public bool do_white_balance_;
        public bool show_help_;

        Model model_;
        //unsigned int program_;
        //GLuint full_screen_quad_vao_;
        //GLuint full_screen_quad_vbo_;
        //std::unique_ptr<TextRenderer> text_renderer_;
        //int window_id_;

        double view_distance_meters_;
        double view_zenith_angle_radians_;
        double view_azimuth_angle_radians_;
        double sun_zenith_angle_radians_;
        double sun_azimuth_angle_radians_;
        double exposure_;

        int previous_mouse_x_;
        int previous_mouse_y_;
        bool is_ctrl_key_pressed_;

        void Start() {
            InitModel();

        }

        void InitModel() {

        }

        void Update() {

        }
    }
}
