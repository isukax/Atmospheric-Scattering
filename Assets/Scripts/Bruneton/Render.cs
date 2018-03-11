using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Bruneton {
    public class Render : MonoBehaviour {

        [Header("Atmosphere Parameters")]
        public bool m_IsUseOzone;
        public bool m_IsHalfPrecision;
        public bool m_IsValidGroudIrradiance;

        [SerializeField]
        Color m_GroundAlbedo;

        [SerializeField, Range(1, 10)]
        uint m_ScatteringOrder = 2;

        [SerializeField, Range(-1, 1)]
        float m_MiePhaseFunctionG = 0.96f;

        [SerializeField, Range(0, 0.1f)]
        float m_SunAngularRadius = 0.00935f / 2.0f;

        float kBottomRadius = 6360000.0f;
        float kTopRadius = 6420000.0f;
        float kLengthUnitInMeters = 1000;
        float kScaleHeightRayleigh = 8000;
        float kScaleHeightMie = 1200;

        [SerializeField, Range(0, 1)]
        float m_DepthTest = 0;

        static readonly int READ = 0;
        static readonly int WRITE = 1;

        public Light m_Sun;
        public Material m_PreComputeTransmittance;
        public ComputeShader m_ComputeShader;

        public RenderTexture m_TransmittanceTexture;

        public RenderTexture[] m_IrradianceTexture;
        public RenderTexture m_DeltaIrradianceTexture;

        public RenderTexture[] m_ScatteringTexture;
        public RenderTexture m_SingleRayleighScatteringTexture;
        public RenderTexture m_SingleMieScatteringTexture;
        public RenderTexture m_ScatteringDensityTexture;
        public RenderTexture m_DeltaMultipleScatteringTexture;

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


        void UpdateMaterialParams(Material mat) {
            Vector3 skySpectralRadianceToLuminance = new Vector3(114974.914f, 71305.95f, 65310.5469f);
            Vector3 sunSpectralRadianceToLuminance = new Vector3(98242.79f, 69954.4f, 66475.0156f);
            mat.SetMatrix("_InverseProjection", Camera.main.projectionMatrix.inverse);
            mat.SetMatrix("_InverseView", Camera.main.cameraToWorldMatrix.inverse);
            mat.SetVector("SKY_SPECTRAL_RADIANCE_TO_LUMINANCE", skySpectralRadianceToLuminance);
            mat.SetVector("SUN_SPECTRAL_RADIANCE_TO_LUMINANCE", sunSpectralRadianceToLuminance);
            mat.SetColor("_GroundAlbedo", m_GroundAlbedo);
            if(m_IsUseOzone) {
                mat.SetVector("_AbsorptionExtinction", new Vector4(0.000650f, 0.001881f, 0.000085f, 0));
            } else {
                mat.SetVector("_AbsorptionExtinction", Vector4.zero);
            }
            mat.SetFloat("_MiePhaseFunctionG", m_MiePhaseFunctionG);
            mat.SetFloat("_TopRadius", kTopRadius / kLengthUnitInMeters);
            mat.SetFloat("_BottomRadius", kBottomRadius / kLengthUnitInMeters);
            mat.SetFloat("_SunAngularRadius", m_SunAngularRadius);
            //mat.SetFloat("exposure", UseLuminance != LUMINANCE.NONE ? Exposure * 1e-5f : Exposure);
            mat.SetVector("earth_center", new Vector3(0.0f, -kBottomRadius / kLengthUnitInMeters, 0.0f));
            mat.SetVector("sun_size", new Vector2(Mathf.Tan(m_SunAngularRadius), Mathf.Cos(m_SunAngularRadius)));
            mat.SetVector("sun_direction", m_Sun.transform.forward * -1.0f);

            UpdateDensityLayer(new List<float>() { 0, 0, 0, 0, 0 }, (name, num) => {
                mat.SetFloat("rayleigh0_" + name, num);
            });
            UpdateDensityLayer(new List<float>() { 0, 1, -1.0f / kScaleHeightRayleigh * kLengthUnitInMeters, 0, 0 }, (name, num) => {
                mat.SetFloat("rayleigh1_" + name, num);
            });
            UpdateDensityLayer(new List<float>() { 0, 0, 0, 0, 0 }, (name, num) => {
                mat.SetFloat("mie0_" + name, num);
            });
            UpdateDensityLayer(new List<float>() { 0, 1, -1.0f / kScaleHeightMie * kLengthUnitInMeters, 0, 0 }, (name, num) => {
                mat.SetFloat("mie1_" + name, num);
            });
            UpdateDensityLayer(new List<float>() { 25, 0, 0, 0.066667f, -0.666667f }, (name, num) => {
                mat.SetFloat("absorption0_" + name, num);
            });
            UpdateDensityLayer(new List<float>() { 25, 0, 0, -0.066667f, 2.666667f }, (name, num) => {
                mat.SetFloat("absorption1_" + name, num);
            });
            double white_point_r = 1.0;
            double white_point_g = 1.0;
            double white_point_b = 1.0;
            //if(DoWhiteBalance) {
            //    m_model.ConvertSpectrumToLinearSrgb(out white_point_r, out white_point_g, out white_point_b);

            //    double white_point = (white_point_r + white_point_g + white_point_b) / 3.0;
            //    white_point_r /= white_point;
            //    white_point_g /= white_point;
            //    white_point_b /= white_point;
            //}

            mat.SetVector("white_point", new Vector3((float)white_point_r, (float)white_point_g, (float)white_point_b));

        }

        void Swap(ref RenderTexture[] texture) {
            RenderTexture temp = texture[READ];
            texture[READ] = texture[WRITE];
            texture[WRITE] = temp;
        }

        void UpdateDensityLayer(List<float> layer, UnityEngine.Events.UnityAction<string, float> action) {
            int length = 5;
            string[] paramName = {"width", "exp_term", "exp_scale", "linear_term", "constant_term" };
            for(int i = 0; i < length; ++i) {
                //mat.SetFloat(paramName[i], layer[i]);
                action(paramName[i], layer[i]);
            }
        }

        void Update() {
            if(Input.GetKeyDown(KeyCode.Space) && m_ComputeShader != null) {
                Debug.Log("dispatch");
                Start();
                if(m_IsUseOzone) {
                    m_ComputeShader.SetVector("_AbsorptionExtinction", new Vector4(0.000650f, 0.001881f, 0.000085f, 0));
                } else {
                    m_ComputeShader.SetVector("_AbsorptionExtinction", Vector4.zero);
                }
                m_ComputeShader.SetVector("_GroundAlbedo", m_GroundAlbedo);
                m_ComputeShader.SetFloat("_MiePhaseFunctionG", m_MiePhaseFunctionG);
                m_ComputeShader.SetFloat("_TopRadius", kTopRadius / kLengthUnitInMeters);
                m_ComputeShader.SetFloat("_BottomRadius", kBottomRadius / kLengthUnitInMeters);
                m_ComputeShader.SetFloat("_SunAngularRadius", m_SunAngularRadius);
                UpdateDensityLayer(new List<float>() { 0, 0, 0, 0, 0 }, (name, num) => {
                    m_ComputeShader.SetFloat("rayleigh0_" + name, num);
                });
                UpdateDensityLayer(new List<float>() { 0, 1, -1.0f / kScaleHeightRayleigh * kLengthUnitInMeters, 0, 0 }, (name, num) => {
                    m_ComputeShader.SetFloat("rayleigh1_" + name, num);
                });
                UpdateDensityLayer(new List<float>() { 0, 0, 0, 0, 0 }, (name, num) => {
                    m_ComputeShader.SetFloat("mie0_" + name, num);
                });
                UpdateDensityLayer(new List<float>() { 0, 1, -1.0f / kScaleHeightMie * kLengthUnitInMeters, 0, 0 }, (name, num) => {
                    m_ComputeShader.SetFloat("mie1_" + name, num);
                });
                UpdateDensityLayer(new List<float>() { 25, 0, 0, 0.066667f, -0.666667f }, (name, num) => {
                    m_ComputeShader.SetFloat("absorption0_" + name, num);
                });
                UpdateDensityLayer(new List<float>() { 25, 0, 0, -0.066667f, 2.666667f }, (name, num) => {
                    m_ComputeShader.SetFloat("absorption1_" + name, num);
                });

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
                m_PreComputeTransmittance.SetTexture("_DeltaMultipleScatteringTex", m_DeltaMultipleScatteringTexture);
                m_PreComputeTransmittance.SetFloat("_DepthTest", m_DepthTest);

                RenderSettings.skybox.SetTexture("_TransmittanceTex", m_TransmittanceTexture);
                RenderSettings.skybox.SetTexture("_ScatteringTex", m_SingleRayleighScatteringTexture);
                RenderSettings.skybox.SetTexture("_MieScatteringTex", m_SingleMieScatteringTexture);
                RenderSettings.skybox.SetTexture("_ScatteringDensityTex", m_ScatteringDensityTexture);
                RenderSettings.skybox.SetTexture("_MieScatteringTex", m_SingleMieScatteringTexture);
                RenderSettings.skybox.SetMatrix("_InverseProjection", Camera.main.projectionMatrix.inverse);
                RenderSettings.skybox.SetMatrix("_InverseView", Camera.main.cameraToWorldMatrix);

                RenderSettings.skybox.SetTexture("transmittance_texture", m_TransmittanceTexture);
                RenderSettings.skybox.SetTexture("_DeltaIrradianceTex", m_DeltaIrradianceTexture);
                RenderSettings.skybox.SetTexture("irradiance_texture", m_IrradianceTexture[READ]);
                RenderSettings.skybox.SetTexture("scattering_texture", m_ScatteringTexture[READ]);
                RenderSettings.skybox.SetTexture("_SingleRayleighScatteringTex", m_SingleRayleighScatteringTexture);
                RenderSettings.skybox.SetTexture("single_mie_scattering_texture", m_SingleMieScatteringTexture);
                RenderSettings.skybox.SetTexture("_ScatteringDensityTex", m_ScatteringDensityTexture);
                RenderSettings.skybox.SetTexture("_DeltaMultipleScatteringTex", m_DeltaMultipleScatteringTexture);
                RenderSettings.skybox.SetFloat("_DepthTest", m_DepthTest);
            }
            m_PreComputeTransmittance.SetFloat("_DepthTest", m_DepthTest);
            UpdateMaterialParams(m_PreComputeTransmittance);
            UpdateMaterialParams(RenderSettings.skybox);
        }

        //void OnRenderImage(RenderTexture src, RenderTexture dest) {
        //    Graphics.Blit(src, dest, m_PreComputeTransmittance);
        //    //Graphics.Blit(src, dest);
        //    //Graphics.Blit(null, m_TransmittanceTexture, m_PreComputeTransmittance);
        //    //Graphics.Blit(m_TransmittanceTexture, dest);
        //}
    }
}
