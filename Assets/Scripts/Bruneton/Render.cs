using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Bruneton {
    public class Render : MonoBehaviour {

        public Material m_PreComputeTransmittance;
        //public ComputeShader m_TransmittanceComputesShader;
        public RenderTexture m_TransmittanceTexture;
        public RenderTexture m_SingleScatteringTexture;
        public ComputeShader m_ComputeShader;

        public float m_DepthTest = 0;

        public Renderer m_modelRenderer;

        // Use this for initialization
        void Start() {
            {
                m_TransmittanceTexture = new RenderTexture(Constants.TRANSMITTANCE_TEXTURE_WIDTH, Constants.TRANSMITTANCE_TEXTURE_HEIGHT, 0, RenderTextureFormat.ARGBFloat);
                m_TransmittanceTexture.dimension = UnityEngine.Rendering.TextureDimension.Tex2D;
                m_TransmittanceTexture.enableRandomWrite = true;
                m_TransmittanceTexture.wrapMode = TextureWrapMode.Clamp;
                m_TransmittanceTexture.useMipMap = false;
                m_TransmittanceTexture.filterMode = FilterMode.Bilinear;
                m_TransmittanceTexture.Create();
            }

            {
                m_SingleScatteringTexture = new RenderTexture(Constants.SCATTERING_TEXTURE_WIDTH, Constants.SCATTERING_TEXTURE_HEIGHT, 0, RenderTextureFormat.ARGBFloat);
                m_SingleScatteringTexture.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
                m_SingleScatteringTexture.enableRandomWrite = true;
                m_SingleScatteringTexture.wrapMode = TextureWrapMode.Clamp;
                m_SingleScatteringTexture.useMipMap = false;
                m_SingleScatteringTexture.filterMode = FilterMode.Bilinear;
                m_SingleScatteringTexture.volumeDepth = Constants.SCATTERING_TEXTURE_DEPTH;
                m_SingleScatteringTexture.Create();
            }

            if(m_PreComputeTransmittance == null) m_PreComputeTransmittance = new Material(Shader.Find("Bruneton/PrecomputeTransmittance"));
        }

        // Update is called once per frame
        void UpdateMaterialParams() {
            //m_PreComputeTransmittance.SetFloat();
            //m_PreComputeTransmittance.
    //            const AtmosphereParameters ATMOSPHERE = AtmosphereParameters(
    //to_string(solar_irradiance, lambdas, 1.0),
    //std::to_string(sun_angular_radius),
    //std::to_string(bottom_radius / length_unit_in_meters),
    //std::to_string(top_radius / length_unit_in_meters),
    //density_profile(rayleigh_density),
    //to_string(
    //    rayleigh_scattering, lambdas, length_unit_in_meters),
    //density_profile(mie_density),
    //to_string(mie_scattering, lambdas, length_unit_in_meters),
    //to_string(mie_extinction, lambdas, length_unit_in_meters),
    //std::to_string(mie_phase_function_g),
    //density_profile(absorption_density),
    //to_string(
    //    absorption_extinction, lambdas, length_unit_in_meters),
    //to_string(ground_albedo, lambdas, 1.0),
    //std::to_string(cos(max_sun_zenith_angle)));
    //    }
        }

        void Update() {
            if(Input.GetKeyDown(KeyCode.Space) && m_ComputeShader != null) {
                Debug.Log("dispatch");
                var kernel = m_ComputeShader.FindKernel("Transmittance");
                m_ComputeShader.SetTexture(kernel, "TransmittanceTex", m_TransmittanceTexture);
                m_ComputeShader.Dispatch(kernel, 32, 8, 1);

                kernel = m_ComputeShader.FindKernel("SingleScattering");
                m_ComputeShader.SetTexture(kernel, "ScatteringTex", m_SingleScatteringTexture);
                m_ComputeShader.SetTexture(kernel, "TransmittanceTexRead", m_TransmittanceTexture);
                m_ComputeShader.Dispatch(kernel, 32, 16, 4);

                m_PreComputeTransmittance.SetTexture("_TransmittanceTex", m_TransmittanceTexture);
                m_PreComputeTransmittance.SetTexture("_ScatteringTex", m_SingleScatteringTexture);
                m_PreComputeTransmittance.SetFloat("_DepthTest", m_DepthTest);
                m_modelRenderer.material.SetTexture("_TransmittanceTex", m_TransmittanceTexture);
                m_modelRenderer.material.SetTexture("_ScatteringTex", m_SingleScatteringTexture);
            }
                m_PreComputeTransmittance.SetFloat("_DepthTest", m_DepthTest);
            m_modelRenderer.material.SetFloat("_DepthTest", m_DepthTest);
        }

        void OnRenderImage(RenderTexture src, RenderTexture dest) {
            UpdateMaterialParams();
            Graphics.Blit(src, dest, m_PreComputeTransmittance);
            //Graphics.Blit(src, dest);
            //Graphics.Blit(null, m_TransmittanceTexture, m_PreComputeTransmittance);
            //Graphics.Blit(m_TransmittanceTexture, dest);
        }
    }
}
