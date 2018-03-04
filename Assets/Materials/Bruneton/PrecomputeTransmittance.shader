Shader "Bruneton/PrecomputeTransmittance"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		_AtmosphereTopRadius("AtmosphereTopRadius", Float) = 6420.0
		_AtmosphereBottomRadius("AtmosphereTopRadius", Float) = 6360.0
		_MiePhaseG("Mie Phase G", Float) = 0.8


	}
	
	CGINCLUDE
	#include "UnityCG.cginc"
	#include "Definitions.cginc"

	//static const int TRANSMITTANCE_TEXTURE_WIDTH = 256;
	//static const int TRANSMITTANCE_TEXTURE_HEIGHT = 64;

	static const float3 kRayleighScatteringCoef = float3(0.005802, 0.013558, 0.0331);
	static const float3 kMieScatteringCoef = float3(0.003996, 00.003996, 0.003996);
	static const float3 kMieExtinctionCoef = float3(0.00444, 0.00444, 0.00444);
	static const float kRayleighScaleHeight = 8.0;	// km
	static const float kMieScaleHeight = 1.2;
	static const float kPi = 3.1415926;

	float _AtmosphereTopRadius;
	float _AtmosphereBottomRadius;
	float _MiePhaseG;
	
	sampler2D _TransmittanceTex;
	sampler2D _DeltaIrradianceTex;
	sampler2D _IrradianceTex;
	sampler3D _ScatteringTex;
	sampler3D _SingleRayleighScatteringTex;
	sampler3D _SingleMieScatteringTex;
	sampler3D _ScatteringDensityTex;
	sampler3D _MultipleScatteringTex;
	sampler3D _DeltaMultipleScatteringTex;
	float _DepthTest;

	struct appdata
	{
		float4 vertex : POSITION;
		float2 uv : TEXCOORD0;
	};

	struct v2f
	{
		float2 uv : TEXCOORD0;
		float4 vertex : SV_POSITION;
	};

	v2f vert (appdata v)
	{
		v2f o;
		o.vertex = UnityObjectToClipPos(v.vertex);
		o.uv = v.uv;
		return o;
	}


	Number ClampCosine(Number mu) {
		return clamp(mu, Number(-1.0), Number(1.0));
	}

	Length ClampDistance(Length d) {
		return max(d, 0.0 * m);
	}

	Length ClampRadius(IN(AtmosphereParameters) atmosphere, Length r) {
		return clamp(r, atmosphere.bottom_radius, atmosphere.top_radius);
	}

	Length SafeSqrt(Area a) {
		return sqrt(max(a, 0.0 * m2));
	}

	Length DistanceToTopAtmosphereBoundary(IN(AtmosphereParameters) atmosphere, Length r, Number mu) {
		Area discriminant = r * r * (mu * mu - 1.0) +
			atmosphere.top_radius * atmosphere.top_radius;
		return ClampDistance(-r * mu + SafeSqrt(discriminant));
	}

	Length DistanceToBottomAtmosphereBoundary(IN(AtmosphereParameters) atmosphere, Length r, Number mu) {
		Area discriminant = r * r * (mu * mu - 1.0) +
			atmosphere.bottom_radius * atmosphere.bottom_radius;
		return ClampDistance(-r * mu - SafeSqrt(discriminant));
	}

	bool RayIntersectsGround(IN(AtmosphereParameters) atmosphere, Length r, Number mu) {
		return mu < 0.0 && r * r * (mu * mu - 1.0) +
			atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0 * m2;
	}

	Number GetLayerDensity(IN(DensityProfileLayer) layer, Length altitude) {
		Number density = layer.exp_term * exp(layer.exp_scale * altitude) +
			layer.linear_term * altitude + layer.constant_term;
		return clamp(density, Number(0.0), Number(1.0));
	}

	Number GetProfileDensity(IN(DensityProfile) profile, Length altitude) {
		return altitude < profile.layers[0].width ?
			GetLayerDensity(profile.layers[0], altitude) :
			GetLayerDensity(profile.layers[1], altitude);
	}

	// 高さrの位置から天頂角mu方向への光学的深度を求める
	Length ComputeOpticalLengthToTopAtmosphereBoundary(IN(AtmosphereParameters) atmosphere, IN(DensityProfile) profile, Length r, Number mu) {
		// Number of intervals for the numerical integration.
		const int SAMPLE_COUNT = 500;
		// The integration step, i.e. the length of each integration interval.
		Length dx =	DistanceToTopAtmosphereBoundary(atmosphere, r, mu) / Number(SAMPLE_COUNT);
		// Integration loop.
		Length result = 0.0 * m;
		for (int i = 0; i <= SAMPLE_COUNT; ++i) {
			Length d_i = Number(i) * dx;
			// Distance between the current sample point and the planet center.
			Length r_i = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
			// Number density at the current sample point (divided by the number density
			// at the bottom of the atmosphere, yielding a dimensionless number).
			Number y_i = GetProfileDensity(profile, r_i - atmosphere.bottom_radius);
			// Sample weight (from the trapezoidal rule).
			Number weight_i = i == 0 || i == SAMPLE_COUNT ? 0.5 : 1.0;
			result += y_i * weight_i * dx;
		}
		return result;
	}

	// レイリー散乱、ミー散乱、吸収の透光学的深度を足し合わせて、expで透過率を求める
	DimensionlessSpectrum ComputeTransmittanceToTopAtmosphereBoundary(IN(AtmosphereParameters) atmosphere, Length r, Number mu) {
		return exp(-(
			atmosphere.rayleigh_scattering *
			ComputeOpticalLengthToTopAtmosphereBoundary(
				atmosphere, atmosphere.rayleigh_density, r, mu) +
			atmosphere.mie_extinction *
			ComputeOpticalLengthToTopAtmosphereBoundary(
				atmosphere, atmosphere.mie_density, r, mu) +
			atmosphere.absorption_extinction *
			ComputeOpticalLengthToTopAtmosphereBoundary(
				atmosphere, atmosphere.absorption_density, r, mu)));
	}

	// 高度rと天頂角muのテクスチャへのマッピング
	Number GetTextureCoordFromUnitRange(Number x, int texture_size) {
		return 0.5 / Number(texture_size) + x * (1.0 - 1.0 / Number(texture_size));
	}

	Number GetUnitRangeFromTextureCoord(Number u, int texture_size) {
		return (u - 0.5 / Number(texture_size)) / (1.0 - 1.0 / Number(texture_size));
	}

	// 水平方向のサンプリングレートを増加させるため、大気終端への距離を考慮
	float2 GetTransmittanceTextureUvFromRMu(IN(AtmosphereParameters) atmosphere, Length r, Number mu) {
		// Distance to top atmosphere boundary for a horizontal ray at ground level.
		Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
			atmosphere.bottom_radius * atmosphere.bottom_radius);
		// Distance to the horizon.
		Length rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
		// Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
		// and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
		Length d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
		Length d_min = atmosphere.top_radius - r;
		Length d_max = rho + H;
		Number x_mu = (d - d_min) / (d_max - d_min);
		Number x_r = rho / H;
		return float2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
			GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
	}

	void GetRMuFromTransmittanceTextureUv(IN(AtmosphereParameters) atmosphere, IN(float2) uv, OUT(Length) r, OUT(Number) mu) {
		Number x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
		Number x_r = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);
		// Distance to top atmosphere boundary for a horizontal ray at ground level.
		Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
			atmosphere.bottom_radius * atmosphere.bottom_radius);
		// Distance to the horizon, from which we can compute r:
		Length rho = H * x_r;
		r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);
		// Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
		// and maximum values over all mu - obtained for (r,1) and (r,mu_horizon) -
		// from which we can recover mu:
		Length d_min = atmosphere.top_radius - r;
		Length d_max = rho + H;
		Length d = d_min + x_mu * (d_max - d_min);
		mu = d == 0.0 * m ? Number(1.0) : (H * H - rho * rho - d * d) / (2.0 * r * d);
		mu = ClampCosine(mu);
	}

	DimensionlessSpectrum ComputeTransmittanceToTopAtmosphereBoundaryTexture(IN(AtmosphereParameters) atmosphere, IN(float2) gl_frag_coord) {
		const float2 TRANSMITTANCE_TEXTURE_SIZE = float2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
		Length r;
		Number mu;
		GetRMuFromTransmittanceTextureUv(atmosphere, gl_frag_coord / TRANSMITTANCE_TEXTURE_SIZE, r, mu);
		//return float3(gl_frag_coord, 0.0);
		return ComputeTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);
	}

	sampler2D _MainTex;
	DensityProfileLayer SetDensityProfileLayer(float width, float exp_term, float exp_scale, float linear_term, float constant_term) {
		DensityProfileLayer layer;
		layer.width = width;
		layer.exp_term = exp_term;
		layer.exp_scale = exp_scale;
		layer.linear_term = linear_term;
		layer.constant_term = constant_term;
		return layer;
	}

	AtmosphereParameters InitAtmosphereParameters() 
	{
		AtmosphereParameters atmosphere;
		atmosphere.solar_irradiance = IrradianceSpectrum(1.474, 1.8504, 1.91198);
		atmosphere.sun_angular_radius = 0.00935 / 2.0;
		atmosphere.bottom_radius = _AtmosphereBottomRadius;
		atmosphere.top_radius = _AtmosphereTopRadius;
		//atmosphere.rayleigh_density = DensityProfile(DensityProfileLayer[2](DensityProfileLayer(0.0, 0.0, 0.0, 0.0, 0.0), DensityProfileLayer(0.0, 1.0, -0.125, 0.0, 0.0)));
		atmosphere.rayleigh_density.layers[0] = SetDensityProfileLayer(0.0, 0.0, 0.0, 0.0, 0.0);
		atmosphere.rayleigh_density.layers[1] = SetDensityProfileLayer(0.0, 1.0, -0.125, 0.0, 0.0);

		atmosphere.rayleigh_scattering = ScatteringSpectrum(0.005802, 0.013558, 0.0331);
		//atmosphere.mie_density = DensityProfile(DensityProfileLayer[2](DensityProfileLayer(0.0, 0.0, 0.0, 0.0, 0.0), DensityProfileLayer(0.0, 1.0, -0.833333, 0.0, 0.0)));
		atmosphere.mie_density.layers[0] = SetDensityProfileLayer(0.0, 0.0, 0.0, 0.0, 0.0);
		atmosphere.mie_density.layers[1] = SetDensityProfileLayer(0.0, 1.0, -0.833333, 0.0, 0.0);

		atmosphere.mie_scattering = ScatteringSpectrum(0.003996, 00.003996, 0.003996);
		atmosphere.mie_extinction = ScatteringSpectrum(0.00444, 0.00444, 0.00444);
		atmosphere.mie_phase_function_g = _MiePhaseG;
		//atmosphere.absorption_density = DensityProfile(DensityProfileLayer[2](DensityProfileLayer(25.0, 0.0, 0.0, 0.066667, -0.0666667), DensityProfileLayer(25.0, 0.0, 0.0, -0.066667, 2.666667)));
		atmosphere.absorption_density.layers[0] = SetDensityProfileLayer(25.0, 0.0, 0.0, 0.066667, -0.666667);
		atmosphere.absorption_density.layers[1] = SetDensityProfileLayer(25.0, 0.0, 0.0, -0.066667, 2.666667);

		atmosphere.absorption_extinction = ScatteringSpectrum(0.000650, 0.001881, 0.000085);
		atmosphere.ground_albedo = ScatteringSpectrum(0.1, 0.1, 0.1);
		atmosphere.mu_s_min =  cos(102.0 / 180.0 * kPi);//-0.207912;
		//const double max_sun_zenith_angle =	(use_half_precision_ ? 102.0 : 120.0) / 180.0 * kPi;
		//constexpr double kSunAngularRadius = 0.00935 / 2.0;

		return atmosphere;
		//struct AtmosphereParameters {
		//	IrradianceSpectrum solar_irradiance;
		//	Angle sun_angular_radius;
		//	Length bottom_radius;
		//	Length top_radius;
		//	DensityProfile rayleigh_density;
		//	ScatteringSpectrum rayleigh_scattering;
		//	DensityProfile mie_density;
		//	ScatteringSpectrum mie_scattering;
		//	ScatteringSpectrum mie_extinction;
		//	Number mie_phase_function_g;
		//	DensityProfile absorption_density;
		//	ScatteringSpectrum absorption_extinction;
		//	DimensionlessSpectrum ground_albedo;
		//	Number mu_s_min;
		//};

	}

	fixed4 frag(v2f i) : SV_Target
	{
		//return tex2D(_TransmittanceTex, i.uv);
		return tex2D(_IrradianceTex, i.uv);
		//return tex2D(_DeltaIrradianceTex, i.uv);
		//return tex3D(_ScatteringDensityTex, float3(i.uv, _DepthTest));
		//return tex3D(_SingleRayleighScatteringTex, float3(i.uv, _DepthTest));
		//return tex3D(_SingleMieScatteringTex, float3(i.uv, _DepthTest));
		
		return tex3D(_ScatteringTex, float3(i.uv, _DepthTest));
		return tex3D(_MultipleScatteringTex, float3(i.uv, _DepthTest));
		return tex3D(_DeltaMultipleScatteringTex, float3(i.uv, _DepthTest));
	
		//return float4(i.uv, 0, 1);
		AtmosphereParameters atmosphere = InitAtmosphereParameters();
		float3 transmittance = ComputeTransmittanceToTopAtmosphereBoundaryTexture(atmosphere, i.uv);
		return float4(transmittance, 1.0);
	}
	
	ENDCG

	SubShader
	{
		Tags{
			"Queue" = "Transparent"
		}

		Pass{
			Cull Off ZWrite Off ZTest Off

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			ENDCG
		}
	}
}
