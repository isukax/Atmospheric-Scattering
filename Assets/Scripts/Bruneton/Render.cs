using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Bruneton {
    public class Render : MonoBehaviour {

        const int READ = 0;
        const int WRITE = 1;

        public Material m_PreComputeTransmittance;

        public RenderTexture m_TransmittanceTexture;

        public RenderTexture[] m_IrradianceTexture;
        public RenderTexture m_DeltaIrradianceTexture;

        public RenderTexture[] m_ScatteringTexture;
        public RenderTexture m_SingleRayleighScatteringTexture;
        public RenderTexture m_SingleMieScatteringTexture;
        public RenderTexture m_ScatteringDensityTexture;
        public RenderTexture m_MultipleScatteringTexture;
        public RenderTexture m_DeltaMultipleScatteringTexture;

        public ComputeShader m_ComputeShader;

        public float m_DepthTest = 0;
        public int m_ScatteringOrder = 2;
        public Renderer m_modelRenderer;

        void Swap(ref RenderTexture[] texture) {
            RenderTexture temp = texture[READ];
            texture[READ] = texture[WRITE];
            texture[WRITE] = temp;
        }

        // Use this for initialization
        void Start() {
            m_IrradianceTexture = new RenderTexture[2];
            m_ScatteringTexture = new RenderTexture[2];

            {
                m_TransmittanceTexture = new RenderTexture(Constants.TRANSMITTANCE_TEXTURE_WIDTH, Constants.TRANSMITTANCE_TEXTURE_HEIGHT, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
                m_TransmittanceTexture.dimension = UnityEngine.Rendering.TextureDimension.Tex2D;
                m_TransmittanceTexture.enableRandomWrite = true;
                m_TransmittanceTexture.wrapMode = TextureWrapMode.Clamp;
                m_TransmittanceTexture.useMipMap = false;
                m_TransmittanceTexture.filterMode = FilterMode.Bilinear;
                m_TransmittanceTexture.Create();
            }

            for(int i = 0; i < 2; ++i)
            {
                m_IrradianceTexture[i] = new RenderTexture(Constants.IRRADIANCE_TEXTURE_WIDTH, Constants.IRRADIANCE_TEXTURE_HEIGHT, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
                m_IrradianceTexture[i].dimension = UnityEngine.Rendering.TextureDimension.Tex2D;
                m_IrradianceTexture[i].enableRandomWrite = true;
                m_IrradianceTexture[i].wrapMode = TextureWrapMode.Clamp;
                m_IrradianceTexture[i].useMipMap = false;
                m_IrradianceTexture[i].filterMode = FilterMode.Bilinear;
                m_IrradianceTexture[i].Create();
            }

            {
                m_DeltaIrradianceTexture = new RenderTexture(Constants.IRRADIANCE_TEXTURE_WIDTH, Constants.IRRADIANCE_TEXTURE_HEIGHT, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
                m_DeltaIrradianceTexture.dimension = UnityEngine.Rendering.TextureDimension.Tex2D;
                m_DeltaIrradianceTexture.enableRandomWrite = true;
                m_DeltaIrradianceTexture.wrapMode = TextureWrapMode.Clamp;
                m_DeltaIrradianceTexture.useMipMap = false;
                m_DeltaIrradianceTexture.filterMode = FilterMode.Bilinear;
                m_DeltaIrradianceTexture.Create();
            }

            for(int i = 0; i < 2; ++i) {
                m_ScatteringTexture[i] = new RenderTexture(Constants.SCATTERING_TEXTURE_WIDTH, Constants.SCATTERING_TEXTURE_HEIGHT, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
                m_ScatteringTexture[i].dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
                m_ScatteringTexture[i].enableRandomWrite = true;
                m_ScatteringTexture[i].wrapMode = TextureWrapMode.Clamp;
                m_ScatteringTexture[i].useMipMap = false;
                m_ScatteringTexture[i].filterMode = FilterMode.Bilinear;
                m_ScatteringTexture[i].volumeDepth = Constants.SCATTERING_TEXTURE_DEPTH;
                m_ScatteringTexture[i].Create();
            }

            {
                m_SingleRayleighScatteringTexture = new RenderTexture(Constants.SCATTERING_TEXTURE_WIDTH, Constants.SCATTERING_TEXTURE_HEIGHT, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
                m_SingleRayleighScatteringTexture.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
                m_SingleRayleighScatteringTexture.enableRandomWrite = true;
                m_SingleRayleighScatteringTexture.wrapMode = TextureWrapMode.Clamp;
                m_SingleRayleighScatteringTexture.useMipMap = false;
                m_SingleRayleighScatteringTexture.filterMode = FilterMode.Bilinear;
                m_SingleRayleighScatteringTexture.volumeDepth = Constants.SCATTERING_TEXTURE_DEPTH;
                m_SingleRayleighScatteringTexture.Create();
            }

            {
                m_SingleMieScatteringTexture = new RenderTexture(Constants.SCATTERING_TEXTURE_WIDTH, Constants.SCATTERING_TEXTURE_HEIGHT, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
                m_SingleMieScatteringTexture.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
                m_SingleMieScatteringTexture.enableRandomWrite = true;
                m_SingleMieScatteringTexture.wrapMode = TextureWrapMode.Clamp;
                m_SingleMieScatteringTexture.useMipMap = false;
                m_SingleMieScatteringTexture.filterMode = FilterMode.Bilinear;
                m_SingleMieScatteringTexture.volumeDepth = Constants.SCATTERING_TEXTURE_DEPTH;
                m_SingleMieScatteringTexture.Create();
            }

            {
                m_MultipleScatteringTexture = new RenderTexture(Constants.SCATTERING_TEXTURE_WIDTH, Constants.SCATTERING_TEXTURE_HEIGHT, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
                m_MultipleScatteringTexture.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
                m_MultipleScatteringTexture.enableRandomWrite = true;
                m_MultipleScatteringTexture.wrapMode = TextureWrapMode.Clamp;
                m_MultipleScatteringTexture.useMipMap = false;
                m_MultipleScatteringTexture.filterMode = FilterMode.Bilinear;
                m_MultipleScatteringTexture.volumeDepth = Constants.SCATTERING_TEXTURE_DEPTH;
                m_MultipleScatteringTexture.Create();
            }

            {
                m_DeltaMultipleScatteringTexture = new RenderTexture(Constants.SCATTERING_TEXTURE_WIDTH, Constants.SCATTERING_TEXTURE_HEIGHT, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
                m_DeltaMultipleScatteringTexture.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
                m_DeltaMultipleScatteringTexture.enableRandomWrite = true;
                m_DeltaMultipleScatteringTexture.wrapMode = TextureWrapMode.Clamp;
                m_DeltaMultipleScatteringTexture.useMipMap = false;
                m_DeltaMultipleScatteringTexture.filterMode = FilterMode.Bilinear;
                m_DeltaMultipleScatteringTexture.volumeDepth = Constants.SCATTERING_TEXTURE_DEPTH;
                m_DeltaMultipleScatteringTexture.Create();
            }

            {
                m_ScatteringDensityTexture = new RenderTexture(Constants.SCATTERING_TEXTURE_WIDTH, Constants.SCATTERING_TEXTURE_HEIGHT, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
                m_ScatteringDensityTexture.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
                m_ScatteringDensityTexture.enableRandomWrite = true;
                m_ScatteringDensityTexture.wrapMode = TextureWrapMode.Clamp;
                m_ScatteringDensityTexture.useMipMap = false;
                m_ScatteringDensityTexture.filterMode = FilterMode.Bilinear;
                m_ScatteringDensityTexture.volumeDepth = Constants.SCATTERING_TEXTURE_DEPTH;
                m_ScatteringDensityTexture.Create();
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
                Start();

                // 透過率
                var kernel = m_ComputeShader.FindKernel("Transmittance");
                // write
                m_ComputeShader.SetTexture(kernel, "TransmittanceTex", m_TransmittanceTexture);
                m_ComputeShader.Dispatch(kernel, 32, 8, 1);

                // 直射照度
                kernel = m_ComputeShader.FindKernel("DirectIrradiance");
                // write
                m_ComputeShader.SetTexture(kernel, "IrradianceTex", m_IrradianceTexture[WRITE]);
                m_ComputeShader.SetTexture(kernel, "DeltaIrradianceTex", m_DeltaIrradianceTexture);
                // read
                m_ComputeShader.SetTexture(kernel, "IrradianceTextureRead", m_IrradianceTexture[READ]);
                m_ComputeShader.SetTexture(kernel, "TransmittanceTexRead", m_TransmittanceTexture);
                m_ComputeShader.Dispatch(kernel, 8, 2, 1);

                Swap(ref m_IrradianceTexture);

                // 単一散乱
                kernel = m_ComputeShader.FindKernel("SingleScattering");
                // write
                m_ComputeShader.SetTexture(kernel, "ScatteringTex", m_ScatteringTexture[WRITE]);
                m_ComputeShader.SetTexture(kernel, "SingleRayleighScatteringTex", m_SingleRayleighScatteringTexture);
                m_ComputeShader.SetTexture(kernel, "SingleMieScatteringTex", m_SingleMieScatteringTexture);
                // read
                m_ComputeShader.SetTexture(kernel, "TransmittanceTexRead", m_TransmittanceTexture);
                m_ComputeShader.Dispatch(kernel, 32, 16, 4);

                Swap(ref m_ScatteringTexture);

                // Compute the 2nd, 3rd and 4th order of scattering, in sequence.
                for(int scattering_order = 2; scattering_order <= m_ScatteringOrder; ++scattering_order) {
                    kernel = m_ComputeShader.FindKernel("ScatteringDensity");
                    // write
                    m_ComputeShader.SetTexture(kernel, "ScatteringDensityTex", m_ScatteringDensityTexture);
                    // read
                    m_ComputeShader.SetTexture(kernel, "SingleRayleighScatteringTextureRead", m_SingleRayleighScatteringTexture);
                    m_ComputeShader.SetTexture(kernel, "SingleMieScatteringTextureRead", m_SingleMieScatteringTexture);
                    m_ComputeShader.SetTexture(kernel, "DeltaMultipleScatteringTextureRead", m_DeltaMultipleScatteringTexture);
                    m_ComputeShader.SetTexture(kernel, "TransmittanceTexRead", m_TransmittanceTexture);
                    m_ComputeShader.SetTexture(kernel, "DeltaIrradianceTextureRead", m_DeltaIrradianceTexture);
                    m_ComputeShader.SetInt("scattering_order", scattering_order);
                    m_ComputeShader.Dispatch(kernel, 32, 16, 4);

                    kernel = m_ComputeShader.FindKernel("IndirectIrradiance");
                    // write
                    m_ComputeShader.SetTexture(kernel, "IrradianceTex", m_IrradianceTexture[WRITE]);
                    m_ComputeShader.SetTexture(kernel, "DeltaIrradianceTex", m_DeltaIrradianceTexture);
                    // read
                    m_ComputeShader.SetTexture(kernel, "IrradianceTextureRead", m_IrradianceTexture[READ]);
                    m_ComputeShader.SetTexture(kernel, "SingleRayleighScatteringTextureRead", m_SingleRayleighScatteringTexture);
                    m_ComputeShader.SetTexture(kernel, "SingleMieScatteringTextureRead", m_SingleMieScatteringTexture);
                    m_ComputeShader.SetTexture(kernel, "DeltaMultipleScatteringTextureRead", m_DeltaMultipleScatteringTexture);
                    m_ComputeShader.SetInt("scattering_order", scattering_order - 1);
                    m_ComputeShader.SetInt("blend", 1);
                    m_ComputeShader.Dispatch(kernel, 8, 2, 1);

                    Swap(ref m_IrradianceTexture);

                    kernel = m_ComputeShader.FindKernel("MultipleScattering");
                    // write
                    m_ComputeShader.SetTexture(kernel, "MultipleScatteringTex", m_ScatteringTexture[WRITE]);
                    m_ComputeShader.SetTexture(kernel, "DeltaMultipleScatteringTex", m_DeltaMultipleScatteringTexture);
                    // read
                    m_ComputeShader.SetTexture(kernel, "TransmittanceTexRead", m_TransmittanceTexture);
                    m_ComputeShader.SetTexture(kernel, "ScatteringDensityTextureRead", m_ScatteringDensityTexture);
                    m_ComputeShader.SetTexture(kernel, "ScatteringTextureRead", m_ScatteringTexture[READ]);
                    m_ComputeShader.SetInt("blend", 1);
                    m_ComputeShader.Dispatch(kernel, 32, 16, 4);

                    Swap(ref m_ScatteringTexture);
                }

                m_PreComputeTransmittance.SetTexture("_TransmittanceTex", m_TransmittanceTexture);
                m_PreComputeTransmittance.SetTexture("_DeltaIrradianceTex", m_DeltaIrradianceTexture);
                m_PreComputeTransmittance.SetTexture("_IrradianceTex", m_IrradianceTexture[READ]);
                m_PreComputeTransmittance.SetTexture("_ScatteringTex", m_ScatteringTexture[READ]);
                m_PreComputeTransmittance.SetTexture("_SingleRayleighScatteringTex", m_SingleRayleighScatteringTexture);
                m_PreComputeTransmittance.SetTexture("_SingleMieScatteringTex", m_SingleMieScatteringTexture);
                m_PreComputeTransmittance.SetTexture("_ScatteringDensityTex", m_ScatteringDensityTexture);
                m_PreComputeTransmittance.SetTexture("_MultipleScatteringTex", m_MultipleScatteringTexture);
                m_PreComputeTransmittance.SetTexture("_DeltaMultipleScatteringTex", m_DeltaMultipleScatteringTexture);
                m_PreComputeTransmittance.SetFloat("_DepthTest", m_DepthTest);

                
                m_modelRenderer.material.SetTexture("_TransmittanceTex", m_TransmittanceTexture);
                m_modelRenderer.material.SetTexture("_ScatteringTex", m_SingleRayleighScatteringTexture);

                RenderSettings.skybox.SetTexture("_TransmittanceTex", m_TransmittanceTexture);
                RenderSettings.skybox.SetTexture("_ScatteringTex", m_SingleRayleighScatteringTexture);
                RenderSettings.skybox.SetTexture("_MieScatteringTex", m_SingleMieScatteringTexture);
                RenderSettings.skybox.SetTexture("_ScatteringDensityTex", m_ScatteringDensityTexture);
                RenderSettings.skybox.SetTexture("_MieScatteringTex", m_SingleMieScatteringTexture);
                RenderSettings.skybox.SetMatrix("_InverseProjection", Camera.main.projectionMatrix.inverse);
                RenderSettings.skybox.SetMatrix("_InverseView", Camera.main.cameraToWorldMatrix);

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
