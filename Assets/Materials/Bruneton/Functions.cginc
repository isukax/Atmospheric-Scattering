/**
* Copyright (c) 2017 Eric Bruneton
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holders nor the names of its
*    contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef BRUNETON_FUNCTIONS_CGINC
#define BRUNETON_FUNCTIONS_CGINC

#include "Definitions.cginc"

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
	Length dx = DistanceToTopAtmosphereBoundary(atmosphere, r, mu) / Number(SAMPLE_COUNT);
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

DensityProfileLayer SetDensityProfileLayer(float width, float exp_term, float exp_scale, float linear_term, float constant_term) {
	DensityProfileLayer layer;
	layer.width = width;
	layer.exp_term = exp_term;
	layer.exp_scale = exp_scale;
	layer.linear_term = linear_term;
	layer.constant_term = constant_term;
	return layer;
}

DimensionlessSpectrum GetTransmittanceToTopAtmosphereBoundary(IN(AtmosphereParameters) atmosphere, IN(TransmittanceTexture) transmittance_texture,	Length r, Number mu) {
	float2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);
	return DimensionlessSpectrum(TEX2D(transmittance_texture, uv).rgb);
}

DimensionlessSpectrum GetTransmittance(IN(AtmosphereParameters) atmosphere, IN(TransmittanceTexture) transmittance_texture, Length r, Number mu, Length d, bool ray_r_mu_intersects_ground) {
	Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
	Number mu_d = ClampCosine((r * mu + d) / r_d);

	if (ray_r_mu_intersects_ground) {
		return min(
			GetTransmittanceToTopAtmosphereBoundary(
				atmosphere, transmittance_texture, r_d, -mu_d) /
			GetTransmittanceToTopAtmosphereBoundary(
				atmosphere, transmittance_texture, r, -mu),
			DimensionlessSpectrum(1.0, 1.0, 1.0));
	}
	else {
		return min(
			GetTransmittanceToTopAtmosphereBoundary(
				atmosphere, transmittance_texture, r, mu) /
			GetTransmittanceToTopAtmosphereBoundary(
				atmosphere, transmittance_texture, r_d, mu_d),
			DimensionlessSpectrum(1.0, 1.0, 1.0));
	}
}

DimensionlessSpectrum GetTransmittanceToSun(IN(AtmosphereParameters) atmosphere, IN(TransmittanceTexture) transmittance_texture, Length r, Number mu_s) {
	Number sin_theta_h = atmosphere.bottom_radius / r;
	Number cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
	return GetTransmittanceToTopAtmosphereBoundary(
		atmosphere, transmittance_texture, r, mu_s) *
		smoothstep(-sin_theta_h * atmosphere.sun_angular_radius / rad,
			sin_theta_h * atmosphere.sun_angular_radius / rad,
			mu_s - cos_theta_h);
}

// 1点におけるレイリー散乱、ミー散乱の透過率
void ComputeSingleScatteringIntegrand(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	Length r, Number mu, Number mu_s, Number nu, Length d,
	bool ray_r_mu_intersects_ground,
	OUT(DimensionlessSpectrum) rayleigh, OUT(DimensionlessSpectrum) mie) {
	Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
	Number mu_s_d = ClampCosine((r * mu_s + d * nu) / r_d);
	DimensionlessSpectrum transmittance =
		GetTransmittance(
			atmosphere, transmittance_texture, r, mu, d,
			ray_r_mu_intersects_ground) *
		GetTransmittanceToSun(
			atmosphere, transmittance_texture, r_d, mu_s_d);
	rayleigh = transmittance * GetProfileDensity(
		atmosphere.rayleigh_density, r_d - atmosphere.bottom_radius);
	mie = transmittance * GetProfileDensity(
		atmosphere.mie_density, r_d - atmosphere.bottom_radius);
}

Length DistanceToNearestAtmosphereBoundary(IN(AtmosphereParameters) atmosphere,
	Length r, Number mu, bool ray_r_mu_intersects_ground) {
	if (ray_r_mu_intersects_ground) {
		return DistanceToBottomAtmosphereBoundary(atmosphere, r, mu);
	}
	else {
		return DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
	}
}

void ComputeSingleScattering(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	Length r, Number mu, Number mu_s, Number nu,
	bool ray_r_mu_intersects_ground,
	OUT(IrradianceSpectrum) rayleigh, OUT(IrradianceSpectrum) mie) {

	// Number of intervals for the numerical integration.
	const int SAMPLE_COUNT = 50;
	// The integration step, i.e. the length of each integration interval.
	Length dx =
		DistanceToNearestAtmosphereBoundary(atmosphere, r, mu,
			ray_r_mu_intersects_ground) / Number(SAMPLE_COUNT);
	// Integration loop.
	DimensionlessSpectrum rayleigh_sum = DimensionlessSpectrum(0.0, 0.0, 0.0);
	DimensionlessSpectrum mie_sum = DimensionlessSpectrum(0.0, 0.0, 0.0);
	for (int i = 0; i <= SAMPLE_COUNT; ++i) {
		Length d_i = Number(i) * dx;
		// The Rayleigh and Mie single scattering at the current sample point.
		DimensionlessSpectrum rayleigh_i;
		DimensionlessSpectrum mie_i;
		ComputeSingleScatteringIntegrand(atmosphere, transmittance_texture,
			r, mu, mu_s, nu, d_i, ray_r_mu_intersects_ground, rayleigh_i, mie_i);
		// Sample weight (from the trapezoidal rule).
		Number weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
		rayleigh_sum += rayleigh_i * weight_i;
		mie_sum += mie_i * weight_i;
	}
	// 散乱係数、放射照度は定数なので最後にかける
	// 位相関数は良い角度の精度のためレンダリング時に適用する
	rayleigh = rayleigh_sum * dx * atmosphere.solar_irradiance *
		atmosphere.rayleigh_scattering;
	mie = mie_sum * dx * atmosphere.solar_irradiance * atmosphere.mie_scattering;
}

// 位相関数
InverseSolidAngle RayleighPhaseFunction(Number nu) {
	InverseSolidAngle k = 3.0 / (16.0 * PI * sr);
	return k * (1.0 + nu * nu);
}

InverseSolidAngle MiePhaseFunction(Number g, Number nu) {
	InverseSolidAngle k = 3.0 / (8.0 * PI * sr) * (1.0 - g * g) / (2.0 + g * g);
	return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}

// 4Dテクスチャへマッピング値を求める変換関数
float4 GetScatteringTextureUvwzFromRMuMuSNu(IN(AtmosphereParameters) atmosphere,
	Length r, Number mu, Number mu_s, Number nu,
	bool ray_r_mu_intersects_ground) {

	// Distance to top atmosphere boundary for a horizontal ray at ground level.
	Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
		atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the horizon.
	Length rho =
		SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
	Number u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);

	// Discriminant of the quadratic equation for the intersections of the ray
	// (r,mu) with the ground (see RayIntersectsGround).
	Length r_mu = r * mu;
	Area discriminant =
		r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;
	Number u_mu;
	if (ray_r_mu_intersects_ground) {
		// Distance to the ground for the ray (r,mu), and its minimum and maximum
		// values over all mu - obtained for (r,-1) and (r,mu_horizon).
		Length d = -r_mu - SafeSqrt(discriminant);
		Length d_min = r - atmosphere.bottom_radius;
		Length d_max = rho;
		u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 :
			(d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
	}
	else {
		// Distance to the top atmosphere boundary for the ray (r,mu), and its
		// minimum and maximum values over all mu - obtained for (r,1) and
		// (r,mu_horizon).
		Length d = -r_mu + SafeSqrt(discriminant + H * H);
		Length d_min = atmosphere.top_radius - r;
		Length d_max = rho + H;
		u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
			(d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
	}

	Length d = DistanceToTopAtmosphereBoundary(
		atmosphere, atmosphere.bottom_radius, mu_s);
	Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;
	Length d_max = H;
	Number a = (d - d_min) / (d_max - d_min);
	Number A =
		-2.0 * atmosphere.mu_s_min * atmosphere.bottom_radius / (d_max - d_min);
	Number u_mu_s = GetTextureCoordFromUnitRange(
		max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

	Number u_nu = (nu + 1.0) / 2.0;
	return float4(u_nu, u_mu_s, u_mu, u_r);
}

void GetRMuMuSNuFromScatteringTextureUvwz(IN(AtmosphereParameters) atmosphere,
	IN(float4) uvwz, OUT(Length) r, OUT(Number) mu, OUT(Number) mu_s,
	OUT(Number) nu, OUT(bool) ray_r_mu_intersects_ground) {

	// Distance to top atmosphere boundary for a horizontal ray at ground level.
	Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
		atmosphere.bottom_radius * atmosphere.bottom_radius);
	// Distance to the horizon.
	Length rho =
		H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);
	r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);

	if (uvwz.z < 0.5) {
		// Distance to the ground for the ray (r,mu), and its minimum and maximum
		// values over all mu - obtained for (r,-1) and (r,mu_horizon) - from which
		// we can recover mu:
		Length d_min = r - atmosphere.bottom_radius;
		Length d_max = rho;
		Length d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
			1.0 - 2.0 * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2);
		mu = d == 0.0 * m ? Number(-1.0) :
			ClampCosine(-(rho * rho + d * d) / (2.0 * r * d));
		ray_r_mu_intersects_ground = true;
	}
	else {
		// Distance to the top atmosphere boundary for the ray (r,mu), and its
		// minimum and maximum values over all mu - obtained for (r,1) and
		// (r,mu_horizon) - from which we can recover mu:
		Length d_min = atmosphere.top_radius - r;
		Length d_max = rho + H;
		Length d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
			2.0 * uvwz.z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2);
		mu = d == 0.0 * m ? Number(1.0) :
			ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));
		ray_r_mu_intersects_ground = false;
	}

	Number x_mu_s =
		GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);
	Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;
	Length d_max = H;
	Number A =
		-2.0 * atmosphere.mu_s_min * atmosphere.bottom_radius / (d_max - d_min);
	Number a = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
	Length d = d_min + min(a, A) * (d_max - d_min);
	mu_s = d == 0.0 * m ? Number(1.0) :
		ClampCosine((H * H - d * d) / (2.0 * atmosphere.bottom_radius * d));

	nu = ClampCosine(uvwz.x * 2.0 - 1.0);
}

float mod(float x, float val) { return x - floor(x * 1.0f / val) * val; }

void GetRMuMuSNuFromScatteringTextureFragCoord(
	IN(AtmosphereParameters) atmosphere, IN(float3) gl_frag_coord,
	OUT(Length) r, OUT(Number) mu, OUT(Number) mu_s, OUT(Number) nu,
	OUT(bool) ray_r_mu_intersects_ground) {
	const float4 SCATTERING_TEXTURE_SIZE = float4(
		SCATTERING_TEXTURE_NU_SIZE - 1,
		SCATTERING_TEXTURE_MU_S_SIZE,
		SCATTERING_TEXTURE_MU_SIZE,
		SCATTERING_TEXTURE_R_SIZE);
	Number frag_coord_nu =
		floor(gl_frag_coord.x / Number(SCATTERING_TEXTURE_MU_S_SIZE));
	Number frag_coord_mu_s =
		fmod(gl_frag_coord.x, Number(SCATTERING_TEXTURE_MU_S_SIZE));
	float4 uvwz =
		float4(frag_coord_nu, frag_coord_mu_s, gl_frag_coord.y, gl_frag_coord.z) /
		SCATTERING_TEXTURE_SIZE;
	GetRMuMuSNuFromScatteringTextureUvwz(
		atmosphere, uvwz, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	// Clamp nu to its valid range of values, given mu and mu_s.
	nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),
		mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)));
}

void ComputeSingleScatteringTexture(IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture, IN(float3) gl_frag_coord,
	OUT(IrradianceSpectrum) rayleigh, OUT(IrradianceSpectrum) mie) {
	Length r;
	Number mu;
	Number mu_s;
	Number nu;
	bool ray_r_mu_intersects_ground;
	GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, gl_frag_coord,
		r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	ComputeSingleScattering(atmosphere, transmittance_texture,
		r, mu, mu_s, nu, ray_r_mu_intersects_ground, rayleigh, mie);
}

TEMPLATE(AbstractSpectrum)
AbstractSpectrum GetScattering(
	IN(AtmosphereParameters) atmosphere,
	IN(AbstractScatteringTexture TEMPLATE_ARGUMENT(AbstractSpectrum))
	scattering_texture,
	Length r, Number mu, Number mu_s, Number nu,
	bool ray_r_mu_intersects_ground) {
	float4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
		atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	Number tex_coord_x = uvwz.x * Number(SCATTERING_TEXTURE_NU_SIZE - 1);
	Number tex_x = floor(tex_coord_x);
	Number lerp = tex_coord_x - tex_x;
	float3 uvw0 = float3((tex_x + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
		uvwz.z, uvwz.w);
	float3 uvw1 = float3((tex_x + 1.0 + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
		uvwz.z, uvwz.w);
	return AbstractSpectrum(TEX3D(scattering_texture, uvw0).rgb * (1.0 - lerp) +
		TEX3D(scattering_texture, uvw1).rgb * lerp);
}

RadianceSpectrum GetScattering(
	IN(AtmosphereParameters) atmosphere,
	IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	IN(ScatteringTexture) multiple_scattering_texture,
	Length r, Number mu, Number mu_s, Number nu,
	bool ray_r_mu_intersects_ground,
	int scattering_order) {
	if (scattering_order == 1) {
		IrradianceSpectrum rayleigh = GetScattering(
			atmosphere, single_rayleigh_scattering_texture, r, mu, mu_s, nu,
			ray_r_mu_intersects_ground);
		IrradianceSpectrum mie = GetScattering(
			atmosphere, single_mie_scattering_texture, r, mu, mu_s, nu,
			ray_r_mu_intersects_ground);
		return rayleigh * RayleighPhaseFunction(nu) +
			mie * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
	}
	else {
		return GetScattering(
			atmosphere, multiple_scattering_texture, r, mu, mu_s, nu,
			ray_r_mu_intersects_ground);
	}
}

// ----------
// Multiple Scattering
// ----------

IrradianceSpectrum GetIrradiance(
	IN(AtmosphereParameters) atmosphere,
	IN(IrradianceTexture) irradiance_texture,
	Length r, Number mu_s);

RadianceDensitySpectrum ComputeScatteringDensity(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	IN(ScatteringTexture) multiple_scattering_texture,
	IN(IrradianceTexture) irradiance_texture,
	Length r, Number mu, Number mu_s, Number nu, int scattering_order) {

	// Compute unit direction vectors for the zenith, the view direction omega and
	// and the sun direction omega_s, such that the cosine of the view-zenith
	// angle is mu, the cosine of the sun-zenith angle is mu_s, and the cosine of
	// the view-sun angle is nu. The goal is to simplify computations below.
	float3 zenith_direction = float3(0.0, 0.0, 1.0);
	float3 omega = float3(sqrt(1.0 - mu * mu), 0.0, mu);
	Number sun_dir_x = omega.x == 0.0 ? 0.0 : (nu - mu * mu_s) / omega.x;
	Number sun_dir_y = sqrt(max(1.0 - sun_dir_x * sun_dir_x - mu_s * mu_s, 0.0));
	float3 omega_s = float3(sun_dir_x, sun_dir_y, mu_s);

	const int SAMPLE_COUNT = 16;
	const Angle dphi = pi / Number(SAMPLE_COUNT);
	const Angle dtheta = pi / Number(SAMPLE_COUNT);
	RadianceDensitySpectrum rayleigh_mie =
		RadianceDensitySpectrum(
			0.0 * watt_per_cubic_meter_per_sr_per_nm,
			0.0 * watt_per_cubic_meter_per_sr_per_nm,
			0.0 * watt_per_cubic_meter_per_sr_per_nm
		);

	// Nested loops for the integral over all the incident directions omega_i.
	for (int l = 0; l < SAMPLE_COUNT; ++l) {
		Angle theta = (Number(l) + 0.5) * dtheta;
		Number cos_theta = cos(theta);
		Number sin_theta = sin(theta);
		bool ray_r_theta_intersects_ground =
			RayIntersectsGround(atmosphere, r, cos_theta);

		// The distance and transmittance to the ground only depend on theta, so we
		// can compute them in the outer loop for efficiency.
		Length distance_to_ground = 0.0 * m;
		DimensionlessSpectrum transmittance_to_ground = DimensionlessSpectrum(0.0, 0.0, 0.0);
		DimensionlessSpectrum ground_albedo = DimensionlessSpectrum(0.0, 0.0, 0.0);
		if (ray_r_theta_intersects_ground) {
			distance_to_ground =
				DistanceToBottomAtmosphereBoundary(atmosphere, r, cos_theta);
			transmittance_to_ground =
				GetTransmittance(atmosphere, transmittance_texture, r, cos_theta,
					distance_to_ground, true /* ray_intersects_ground */);
			ground_albedo = atmosphere.ground_albedo;
		}

		for (int m = 0; m < 2 * SAMPLE_COUNT; ++m) {
			Angle phi = (Number(m) + 0.5) * dphi;
			float3 omega_i =
				float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
			SolidAngle domega_i = (dtheta / rad) * (dphi / rad) * sin(theta) * sr;

			// The radiance L_i arriving from direction omega_i after n-1 bounces is
			// the sum of a term given by the precomputed scattering texture for the
			// (n-1)-th order:
			Number nu1 = dot(omega_s, omega_i);
			RadianceSpectrum incident_radiance = GetScattering(atmosphere,
				single_rayleigh_scattering_texture, single_mie_scattering_texture,
				multiple_scattering_texture, r, omega_i.z, mu_s, nu1,
				ray_r_theta_intersects_ground, scattering_order - 1);

			// and of the contribution from the light paths with n-1 bounces and whose
			// last bounce is on the ground. This contribution is the product of the
			// transmittance to the ground, the ground albedo, the ground BRDF, and
			// the irradiance received on the ground after n-2 bounces.
			float3 ground_normal =
				normalize(zenith_direction * r + omega_i * distance_to_ground);
			IrradianceSpectrum ground_irradiance = GetIrradiance(
				atmosphere, irradiance_texture, atmosphere.bottom_radius,
				dot(ground_normal, omega_s));
			incident_radiance += transmittance_to_ground *
				ground_albedo * (1.0 / (PI * sr)) * ground_irradiance;

			// The radiance finally scattered from direction omega_i towards direction
			// -omega is the product of the incident radiance, the scattering
			// coefficient, and the phase function for directions omega and omega_i
			// (all this summed over all particle types, i.e. Rayleigh and Mie).
			Number nu2 = dot(omega, omega_i);
			Number rayleigh_density = GetProfileDensity(
				atmosphere.rayleigh_density, r - atmosphere.bottom_radius);
			Number mie_density = GetProfileDensity(
				atmosphere.mie_density, r - atmosphere.bottom_radius);
			rayleigh_mie += incident_radiance * (
				atmosphere.rayleigh_scattering * rayleigh_density *
				RayleighPhaseFunction(nu2) +
				atmosphere.mie_scattering * mie_density *
				MiePhaseFunction(atmosphere.mie_phase_function_g, nu2)) *
				domega_i;
		}
	}
	return rayleigh_mie;
}

RadianceSpectrum ComputeMultipleScattering(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(ScatteringDensityTexture) scattering_density_texture,
	Length r, Number mu, Number mu_s, Number nu,
	bool ray_r_mu_intersects_ground) {

	// Number of intervals for the numerical integration.
	const int SAMPLE_COUNT = 50;
	// The integration step, i.e. the length of each integration interval.
	Length dx =
		DistanceToNearestAtmosphereBoundary(
			atmosphere, r, mu, ray_r_mu_intersects_ground) /
		Number(SAMPLE_COUNT);
	// Integration loop.
	RadianceSpectrum rayleigh_mie_sum =
		RadianceSpectrum(
			0.0 * watt_per_square_meter_per_sr_per_nm,
			0.0 * watt_per_square_meter_per_sr_per_nm,
			0.0 * watt_per_square_meter_per_sr_per_nm
		);
	for (int i = 0; i <= SAMPLE_COUNT; ++i) {
		Length d_i = Number(i) * dx;

		// The r, mu and mu_s parameters at the current integration point (see the
		// single scattering section for a detailed explanation).
		Length r_i =
			ClampRadius(atmosphere, sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r));
		Number mu_i = ClampCosine((r * mu + d_i) / r_i);
		Number mu_s_i = ClampCosine((r * mu_s + d_i * nu) / r_i);

		// The Rayleigh and Mie multiple scattering at the current sample point.
		RadianceSpectrum rayleigh_mie_i =
			GetScattering(
				atmosphere, scattering_density_texture, r_i, mu_i, mu_s_i, nu,
				ray_r_mu_intersects_ground) *
			GetTransmittance(
				atmosphere, transmittance_texture, r, mu, d_i,
				ray_r_mu_intersects_ground) *
			dx;
		// Sample weight (from the trapezoidal rule).
		Number weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
		rayleigh_mie_sum += rayleigh_mie_i * weight_i;
	}
	return rayleigh_mie_sum;
}

RadianceDensitySpectrum ComputeScatteringDensityTexture(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	IN(ScatteringTexture) multiple_scattering_texture,
	IN(IrradianceTexture) irradiance_texture,
	IN(float3) gl_frag_coord, int scattering_order) {
	Length r;
	Number mu;
	Number mu_s;
	Number nu;
	bool ray_r_mu_intersects_ground;
	GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, gl_frag_coord,
		r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	return ComputeScatteringDensity(atmosphere, transmittance_texture,
		single_rayleigh_scattering_texture, single_mie_scattering_texture,
		multiple_scattering_texture, irradiance_texture, r, mu, mu_s, nu,
		scattering_order);
}

RadianceSpectrum ComputeMultipleScatteringTexture(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(ScatteringDensityTexture) scattering_density_texture,
	IN(float3) gl_frag_coord, OUT(Number) nu) {
	Length r;
	Number mu;
	Number mu_s;
	bool ray_r_mu_intersects_ground;
	GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, gl_frag_coord,
		r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	return ComputeMultipleScattering(atmosphere, transmittance_texture,
		scattering_density_texture, r, mu, mu_s, nu,
		ray_r_mu_intersects_ground);
}

// ----------
// Ground Irradiance
// ----------

IrradianceSpectrum ComputeDirectIrradiance(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	Length r, Number mu_s) {

	Number alpha_s = atmosphere.sun_angular_radius / rad;
	// Approximate average of the cosine factor mu_s over the visible fraction of
	// the Sun disc.
	Number average_cosine_factor =
		mu_s < -alpha_s ? 0.0 : (mu_s > alpha_s ? mu_s :
			(mu_s + alpha_s) * (mu_s + alpha_s) / (4.0 * alpha_s));

	return atmosphere.solar_irradiance *
		GetTransmittanceToTopAtmosphereBoundary(
			atmosphere, transmittance_texture, r, mu_s) * average_cosine_factor;

}

IrradianceSpectrum ComputeIndirectIrradiance(
	IN(AtmosphereParameters) atmosphere,
	IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	IN(ScatteringTexture) multiple_scattering_texture,
	Length r, Number mu_s, int scattering_order) {

	const int SAMPLE_COUNT = 32;
	const Angle dphi = pi / Number(SAMPLE_COUNT);
	const Angle dtheta = pi / Number(SAMPLE_COUNT);

	IrradianceSpectrum result =IrradianceSpectrum(
		0.0 * watt_per_square_meter_per_nm,
		0.0 * watt_per_square_meter_per_nm,
		0.0 * watt_per_square_meter_per_nm
	);
	float3 omega_s = float3(sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);
	for (int j = 0; j < SAMPLE_COUNT / 2; ++j) {
		Angle theta = (Number(j) + 0.5) * dtheta;
		for (int i = 0; i < 2 * SAMPLE_COUNT; ++i) {
			Angle phi = (Number(i) + 0.5) * dphi;
			float3 omega =
				float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
			SolidAngle domega = (dtheta / rad) * (dphi / rad) * sin(theta) * sr;

			Number nu = dot(omega, omega_s);
			result += GetScattering(atmosphere, single_rayleigh_scattering_texture,
				single_mie_scattering_texture, multiple_scattering_texture,
				r, omega.z, mu_s, nu, false /* ray_r_theta_intersects_ground */,
				scattering_order) *
				omega.z * domega;
		}
	}
	return result;
}

float2 GetIrradianceTextureUvFromRMuS(IN(AtmosphereParameters) atmosphere,
	Length r, Number mu_s) {
	Number x_r = (r - atmosphere.bottom_radius) /
		(atmosphere.top_radius - atmosphere.bottom_radius);
	Number x_mu_s = mu_s * 0.5 + 0.5;
	return float2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
		GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
}

void GetRMuSFromIrradianceTextureUv(IN(AtmosphereParameters) atmosphere,
	IN(float2) uv, OUT(Length) r, OUT(Number) mu_s) {
	Number x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
	Number x_r = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
	r = atmosphere.bottom_radius +
		x_r * (atmosphere.top_radius - atmosphere.bottom_radius);
	mu_s = ClampCosine(2.0 * x_mu_s - 1.0);
}
//
//static const float2 IRRADIANCE_TEXTURE_SIZE =
//float2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);

IrradianceSpectrum ComputeDirectIrradianceTexture(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(float2) gl_frag_coord) {
	Length r;
	Number mu_s;
	const float2 IRRADIANCE_TEXTURE_SIZE = float2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);

	GetRMuSFromIrradianceTextureUv(
		atmosphere, gl_frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);
	return ComputeDirectIrradiance(atmosphere, transmittance_texture, r, mu_s);
}

IrradianceSpectrum ComputeIndirectIrradianceTexture(
	IN(AtmosphereParameters) atmosphere,
	IN(ReducedScatteringTexture) single_rayleigh_scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	IN(ScatteringTexture) multiple_scattering_texture,
	IN(float2) gl_frag_coord, int scattering_order) {
	Length r;
	Number mu_s;
	const float2 IRRADIANCE_TEXTURE_SIZE = float2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);

	GetRMuSFromIrradianceTextureUv(
		atmosphere, gl_frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);
	return ComputeIndirectIrradiance(atmosphere,
		single_rayleigh_scattering_texture, single_mie_scattering_texture,
		multiple_scattering_texture, r, mu_s, scattering_order);
}

IrradianceSpectrum GetIrradiance(
	IN(AtmosphereParameters) atmosphere,
	IN(IrradianceTexture) irradiance_texture,
	Length r, Number mu_s) {
	float2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);
	return IrradianceSpectrum(TEX2D(irradiance_texture, uv).rgb);
}

// ----------
// Rendering
// ----------

#ifdef COMBINED_SCATTERING_TEXTURES
float3 GetExtrapolatedSingleMieScattering(
	IN(AtmosphereParameters) atmosphere, IN(float4) scattering) {
	if (scattering.r == 0.0) {
		return float3(0.0);
	}
	return scattering.rgb * scattering.a / scattering.r *
		(atmosphere.rayleigh_scattering.r / atmosphere.mie_scattering.r) *
		(atmosphere.mie_scattering / atmosphere.rayleigh_scattering);
}
#endif

IrradianceSpectrum GetCombinedScattering(
	IN(AtmosphereParameters) atmosphere,
	IN(ReducedScatteringTexture) scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	Length r, Number mu, Number mu_s, Number nu,
	bool ray_r_mu_intersects_ground,
	OUT(IrradianceSpectrum) single_mie_scattering) {
	float4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
		atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
	Number tex_coord_x = uvwz.x * Number(SCATTERING_TEXTURE_NU_SIZE - 1);
	Number tex_x = floor(tex_coord_x);
	Number lerp = tex_coord_x - tex_x;
	float3 uvw0 = float3((tex_x + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
		uvwz.z, uvwz.w);
	float3 uvw1 = float3((tex_x + 1.0 + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
		uvwz.z, uvwz.w);
#ifdef COMBINED_SCATTERING_TEXTURES
	float4 combined_scattering =
		TEX3D(scattering_texture, uvw0) * (1.0 - lerp) +
		TEX3D(scattering_texture, uvw1) * lerp;
	IrradianceSpectrum scattering = IrradianceSpectrum(combined_scattering);
	single_mie_scattering =
		GetExtrapolatedSingleMieScattering(atmosphere, combined_scattering);
#else
	IrradianceSpectrum scattering = IrradianceSpectrum(
		TEX3D(scattering_texture, uvw0).rgb * (1.0 - lerp) +
		TEX3D(scattering_texture, uvw1).rgb * lerp);
	single_mie_scattering = IrradianceSpectrum(
		TEX3D(single_mie_scattering_texture, uvw0).rgb * (1.0 - lerp) +
		TEX3D(single_mie_scattering_texture, uvw1).rgb * lerp);
#endif
	return scattering;
}

RadianceSpectrum GetSkyRadiance(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(ReducedScatteringTexture) scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	Position camera, IN(Direction) view_ray, Length shadow_length,
	IN(Direction) sun_direction, OUT(DimensionlessSpectrum) transmittance) {
	// Compute the distance to the top atmosphere boundary along the view ray,
	// assuming the viewer is in space (or NaN if the view ray does not intersect
	// the atmosphere).
	Length r = length(camera);
	Length rmu = dot(camera, view_ray);
	Length distance_to_top_atmosphere_boundary = -rmu -
		sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
	// If the viewer is in space and the view ray intersects the atmosphere, move
	// the viewer to the top atmosphere boundary (along the view ray):
	if (distance_to_top_atmosphere_boundary > 0.0 * m) {
		camera = camera + view_ray * distance_to_top_atmosphere_boundary;
		r = atmosphere.top_radius;
		rmu += distance_to_top_atmosphere_boundary;
	}
	else if (r > atmosphere.top_radius) {
		// If the view ray does not intersect the atmosphere, simply return 0.
		transmittance = DimensionlessSpectrum(1.0, 1.0, 1.0);
		return RadianceSpectrum(
			0.0 * watt_per_square_meter_per_sr_per_nm,
			0.0 * watt_per_square_meter_per_sr_per_nm,
			0.0 * watt_per_square_meter_per_sr_per_nm
		);
	}
	// Compute the r, mu, mu_s and nu parameters needed for the texture lookups.
	Number mu = rmu / r;
	Number mu_s = dot(camera, sun_direction) / r;
	Number nu = dot(view_ray, sun_direction);
	bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

	transmittance = ray_r_mu_intersects_ground ? DimensionlessSpectrum(0.0, 0.0, 0.0) :
		GetTransmittanceToTopAtmosphereBoundary(
			atmosphere, transmittance_texture, r, mu);
	IrradianceSpectrum single_mie_scattering;
	IrradianceSpectrum scattering;
	if (shadow_length == 0.0 * m) {
		scattering = GetCombinedScattering(
			atmosphere, scattering_texture, single_mie_scattering_texture,
			r, mu, mu_s, nu, ray_r_mu_intersects_ground,
			single_mie_scattering);
	}
	else {
		// Case of light shafts (shadow_length is the total length noted l in our
		// paper): we omit the scattering between the camera and the point at
		// distance l, by implementing Eq. (18) of the paper (shadow_transmittance
		// is the T(x,x_s) term, scattering is the S|x_s=x+lv term).
		Length d = shadow_length;
		Length r_p =
			ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
		Number mu_p = (r * mu + d) / r_p;
		Number mu_s_p = (r * mu_s + d * nu) / r_p;

		scattering = GetCombinedScattering(
			atmosphere, scattering_texture, single_mie_scattering_texture,
			r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
			single_mie_scattering);
		DimensionlessSpectrum shadow_transmittance =
			GetTransmittance(atmosphere, transmittance_texture,
				r, mu, shadow_length, ray_r_mu_intersects_ground);
		scattering = scattering * shadow_transmittance;
		single_mie_scattering = single_mie_scattering * shadow_transmittance;
	}
	return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
		MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}

RadianceSpectrum GetSkyRadianceToPoint(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(ReducedScatteringTexture) scattering_texture,
	IN(ReducedScatteringTexture) single_mie_scattering_texture,
	Position camera, IN(Position) points, Length shadow_length,
	IN(Direction) sun_direction, OUT(DimensionlessSpectrum) transmittance) {
	// Compute the distance to the top atmosphere boundary along the view ray,
	// assuming the viewer is in space (or NaN if the view ray does not intersect
	// the atmosphere).
	Direction view_ray = normalize(points - camera);
	Length r = length(camera);
	Length rmu = dot(camera, view_ray);
	Length distance_to_top_atmosphere_boundary = -rmu -
		sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
	// If the viewer is in space and the view ray intersects the atmosphere, move
	// the viewer to the top atmosphere boundary (along the view ray):
	if (distance_to_top_atmosphere_boundary > 0.0 * m) {
		camera = camera + view_ray * distance_to_top_atmosphere_boundary;
		r = atmosphere.top_radius;
		rmu += distance_to_top_atmosphere_boundary;
	}

	// Compute the r, mu, mu_s and nu parameters for the first texture lookup.
	Number mu = rmu / r;
	Number mu_s = dot(camera, sun_direction) / r;
	Number nu = dot(view_ray, sun_direction);
	Length d = length(points - camera);
	bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

	transmittance = GetTransmittance(atmosphere, transmittance_texture,
		r, mu, d, ray_r_mu_intersects_ground);

	IrradianceSpectrum single_mie_scattering;
	IrradianceSpectrum scattering = GetCombinedScattering(
		atmosphere, scattering_texture, single_mie_scattering_texture,
		r, mu, mu_s, nu, ray_r_mu_intersects_ground,
		single_mie_scattering);

	// Compute the r, mu, mu_s and nu parameters for the second texture lookup.
	// If shadow_length is not 0 (case of light shafts), we want to ignore the
	// scattering along the last shadow_length meters of the view ray, which we
	// do by subtracting shadow_length from d (this way scattering_p is equal to
	// the S|x_s=x_0-lv term in Eq. (17) of our paper).
	d = max(d - shadow_length, 0.0 * m);
	Length r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
	Number mu_p = (r * mu + d) / r_p;
	Number mu_s_p = (r * mu_s + d * nu) / r_p;

	IrradianceSpectrum single_mie_scattering_p;
	IrradianceSpectrum scattering_p = GetCombinedScattering(
		atmosphere, scattering_texture, single_mie_scattering_texture,
		r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
		single_mie_scattering_p);

	// Combine the lookup results to get the scattering between camera and point.
	DimensionlessSpectrum shadow_transmittance = transmittance;
	if (shadow_length > 0.0 * m) {
		// This is the T(x,x_s) term in Eq. (17) of our paper, for light shafts.
		shadow_transmittance = GetTransmittance(atmosphere, transmittance_texture,
			r, mu, d, ray_r_mu_intersects_ground);
	}
	scattering = scattering - shadow_transmittance * scattering_p;
	single_mie_scattering =
		single_mie_scattering - shadow_transmittance * single_mie_scattering_p;
#ifdef COMBINED_SCATTERING_TEXTURES
	single_mie_scattering = GetExtrapolatedSingleMieScattering(
		atmosphere, float4(scattering, single_mie_scattering.r));
#endif

	// Hack to avoid rendering artifacts when the sun is below the horizon.
	single_mie_scattering = single_mie_scattering *
		smoothstep(Number(0.0), Number(0.01), mu_s);

	return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
		MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}

IrradianceSpectrum GetSunAndSkyIrradiance(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture,
	IN(IrradianceTexture) irradiance_texture,
	IN(Position) points, IN(Direction) normal, IN(Direction) sun_direction,
	OUT(IrradianceSpectrum) sky_irradiance) {
	Length r = length(points);
	Number mu_s = dot(points, sun_direction) / r;

	// Indirect irradiance (approximated if the surface is not horizontal).
	sky_irradiance = GetIrradiance(atmosphere, irradiance_texture, r, mu_s) *
		(1.0 + dot(normal, points) / r) * 0.5;

	// Direct irradiance.
	return atmosphere.solar_irradiance *
		GetTransmittanceToSun(
			atmosphere, transmittance_texture, r, mu_s) *
		max(dot(normal, sun_direction), 0.0);
}


Angle _SunAngularRadius;
Length _TopRadius;
Length _BottomRadius;
Number _MiePhaseFunctionG;
ScatteringSpectrum _GroundAlbedo;
float4 _AbsorptionExtinction;
//ScatteringSpectrum _AbsorptionExtinction;
float _RayleighLayer0[5];
uniform float _RayleighLayer1[5];
float rayleigh0_width;
float rayleigh0_exp_term;
float rayleigh0_exp_scale;
float rayleigh0_linear_term;
float rayleigh0_constant_term;
float rayleigh1_width;
float rayleigh1_exp_term;
float rayleigh1_exp_scale;
float rayleigh1_linear_term;
float rayleigh1_constant_term;
float mie0_width;
float mie0_exp_term;
float mie0_exp_scale;
float mie0_linear_term;
float mie0_constant_term;
float mie1_width;
float mie1_exp_term;
float mie1_exp_scale;
float mie1_linear_term;
float mie1_constant_term;
float absorption0_width;
float absorption0_exp_term;
float absorption0_exp_scale;
float absorption0_linear_term;
float absorption0_constant_term;
float absorption1_width;
float absorption1_exp_term;
float absorption1_exp_scale;
float absorption1_linear_term;
float absorption1_constant_term;
//// The solar irradiance at the top of the atmosphere.
//IrradianceSpectrum solar_irradiance;
//// The scattering coefficient of air molecules at the altitude where their
//// density is maximum (usually the bottom of the atmosphere), as a function of
//// wavelength. The scattering coefficient at altitude h is equal to
//// 'rayleigh_scattering' times 'rayleigh_density' at this altitude.
//ScatteringSpectrum rayleigh_scattering;
//// The scattering coefficient of aerosols at the altitude where their density
//// is maximum (usually the bottom of the atmosphere), as a function of
//// wavelength. The scattering coefficient at altitude h is equal to
//// 'mie_scattering' times 'mie_density' at this altitude.
//ScatteringSpectrum mie_scattering;
//// The extinction coefficient of aerosols at the altitude where their density
//// is maximum (usually the bottom of the atmosphere), as a function of
//// wavelength. The extinction coefficient at altitude h is equal to
//// 'mie_extinction' times 'mie_density' at this altitude.
//ScatteringSpectrum mie_extinction;
//// The asymetry parameter for the Cornette-Shanks phase function for the
//// aerosols.
//Number mie_phase_function_g;
//// The density profile of air molecules that absorb light (e.g. ozone), i.e.
//// a function from altitude to dimensionless values between 0 (null density)
//// and 1 (maximum density).
//DensityProfile absorption_density;
//// The extinction coefficient of molecules that absorb light (e.g. ozone) at
//// the altitude where their density is maximum, as a function of wavelength.
//// The extinction coefficient at altitude h is equal to
//// 'absorption_extinction' times 'absorption_density' at this altitude.
//ScatteringSpectrum absorption_extinction;
//// The average albedo of the ground.
//DimensionlessSpectrum ground_albedo;
//// The cosine of the maximum Sun zenith angle for which atmospheric scattering
//// must be precomputed (for maximum precision, use the smallest Sun zenith
//// angle yielding negligible sky light radiance values. For instance, for the
//// Earth case, 102 degrees is a good choice - yielding mu_s_min = -0.2).
//Number mu_s_min;

DensityProfileLayer SetDensityProfileLayer(float layerParams[5]) {
	DensityProfileLayer layer;
	layer.width = layerParams[0];
	layer.exp_term = layerParams[1];
	layer.exp_scale = layerParams[2];
	layer.linear_term = layerParams[3];
	layer.constant_term = layerParams[4];
	return layer;
}
AtmosphereParameters InitAtmosphereParameters()
{
	AtmosphereParameters atmosphere;
	atmosphere.solar_irradiance = IrradianceSpectrum(1.474, 1.8504, 1.91198);
	atmosphere.sun_angular_radius = _SunAngularRadius;
	atmosphere.bottom_radius = _BottomRadius;
	atmosphere.top_radius = _TopRadius;

	atmosphere.rayleigh_density.layers[0] = SetDensityProfileLayer(rayleigh0_width, rayleigh0_exp_term, rayleigh0_exp_scale, rayleigh0_linear_term, rayleigh0_constant_term);
	atmosphere.rayleigh_density.layers[1] = SetDensityProfileLayer(rayleigh1_width, rayleigh1_exp_term, rayleigh1_exp_scale, rayleigh1_linear_term, rayleigh1_constant_term);
	atmosphere.rayleigh_scattering = ScatteringSpectrum(0.005802, 0.013558, 0.0331);
	
	atmosphere.mie_density.layers[0] = SetDensityProfileLayer(mie0_width, mie0_exp_term, mie0_exp_scale, mie0_linear_term, mie0_constant_term);
	atmosphere.mie_density.layers[1] = SetDensityProfileLayer(mie1_width, mie1_exp_term, mie1_exp_scale, mie1_linear_term, mie1_constant_term);
	atmosphere.mie_scattering = ScatteringSpectrum(0.003996, 00.003996, 0.003996);
	atmosphere.mie_extinction = ScatteringSpectrum(0.00444, 0.00444, 0.00444);
	atmosphere.mie_phase_function_g = _MiePhaseFunctionG;

	atmosphere.absorption_density.layers[0] = SetDensityProfileLayer(absorption0_width, absorption0_exp_term, absorption0_exp_scale, absorption0_linear_term, absorption0_constant_term);
	atmosphere.absorption_density.layers[1] = SetDensityProfileLayer(absorption1_width, absorption1_exp_term, absorption1_exp_scale, absorption1_linear_term, absorption1_constant_term);

	atmosphere.absorption_extinction = _AbsorptionExtinction;
	//atmosphere.absorption_extinction = ScatteringSpectrum(0.000650, 0.001881, 0.000085);
	atmosphere.ground_albedo = _GroundAlbedo;// ScatteringSpectrum(0.1, 0.1, 0.1);
	atmosphere.mu_s_min = cos(120.0 / 180.0 * PI);//-0.207912;
												   
												  
												  //const double max_sun_zenith_angle =	(use_half_precision_ ? 102.0 : 120.0) / 180.0 * kPi;
												   //atmosphere.sun_angular_radius = 0.00935 / 2.0;
												   //atmosphere.bottom_radius = 6360.0;
												   //atmosphere.top_radius = 6420.0;
												   //atmosphere.rayleigh_density.layers[0] = SetDensityProfileLayer(0.0, 0.0, 0.0, 0.0, 0.0);
												   //atmosphere.rayleigh_density.layers[1] = SetDensityProfileLayer(0.0, 1.0, -0.125, 0.0, 0.0);
												   //atmosphere.mie_density.layers[0] = SetDensityProfileLayer(0.0, 0.0, 0.0, 0.0, 0.0);
												   //atmosphere.mie_density.layers[1] = SetDensityProfileLayer(0.0, 1.0, -0.833333, 0.0, 0.0);
	//atmosphere.absorption_density.layers[0] = SetDensityProfileLayer(25.0, 0.0, 0.0, 0.066667, -0.666667);
	//atmosphere.absorption_density.layers[1] = SetDensityProfileLayer(25.0, 0.0, 0.0, -0.066667, 2.666667);
	//atmosphere.mie_phase_function_g = 0.96;

	return atmosphere;
	//struct AtmosphereParameters {
	//	IrradianceSpectrum solar_irradiance;
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

#endif