Shader "Custom/SkyboxBluneton" 
{
	Properties
	{
		[Toggle(IS_TONEMAP)]_IsToneMap("enable Tonemap", Int) = 0
		_ToneMapExposure("ToneMap Exposure", Range(0, 40)) = 1.3
		[KeywordEnum(Default, Rayleigh, Mie)] _ScatterMode("Scatter Mode", Int) = 0
		_Exposure("Exposure", Range(0, 40)) = 1.3
		_AtmosphereThickness("Atmosphere Thickness", Range(0,5)) = 1.0

		_ScaleHeightR("Scale Height Rayleigh", Float) = 7994
		_ScaleHeightM("Scale Height Mie", Float) = 1200
		_MeanCosine("Mean Cosine(G)", Range(0, 1)) = 0.76

		_ViewDirSampleNum("View Direction Sample Num", Range(1, 1000)) = 16
		_LightDirSampleNum("Light Direction Sample Num", Range(1, 1000)) = 8

		_CameraHeight("Camera Height", Vector) = (0, 0, 0)
	}

	SubShader{
		Pass{
			Tags{ "Queue" = "Background" "RenderType" = "Background" "PreviewType" = "Skybox" }
			Cull Off
			ZWrite Off

			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"
#include "Lighting.cginc"
#include "Functions.cginc"

			#pragma multi_compile _SCATTERMODE_DEFAULT _SCATTERMODE_RAYLEIGH _SCATTERMODE_MIE

			#define SCATTER_MODE_DEFAULT 0
			#define SCATTER_MODE_RAYLEIGH 1
			#define SCATTER_MODE_MIE 2

			#ifndef SCATTER_MODE
			#if defined(_SCATTERMODE_RAYLEIGH)
			#define SCATTER_MODE SCATTER_MODE_RAYLEIGH
			#elif defined(_SCATTERMODE_MIE)
			#define SCATTER_MODE SCATTER_MODE_MIE
			#else
			#define SCATTER_MODE SCATTER_MODE_DEFAULT
			#endif
			#endif

		static const float3 kSphereCenter = float3(0.0, 1.0, 0.0);
	static const float kSphereRadius = 1.0;
	static const float3 kSphereAlbedo = float3(0.8, 0.8, 0.8);
	static const float3 kGroundAlbedo = float3(0.0, 0.0, 0.04);

	float3 white_point;
	float3 earth_center;
	float3 sun_direction;
	float2 sun_size;
	float depth_test;

	float sun_angular_radius;
	float3 SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
	float3 SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;

		sampler2D transmittance_texture;
	sampler2D irradiance_texture;
	sampler3D delta_scattering_density_texture;
	sampler3D scattering_texture;
	sampler3D single_mie_scattering_texture;

			sampler2D _TransmittanceTex;
			sampler3D _ScatteringTex;
			sampler3D _MieScatteringTex;
			float _ToneMapExposure;
			matrix _InverseProjection;
			matrix _InverseView;

			const static float M_PI = 3.141592653;
			const static float kInf = 10e6;

			const static float EarthRadius = 6360e3;
			const static float AtmosphereRadius = 6420e3;

			const static float3 betaR = float3(3.8e-6f, 13.5e-6f, 33.1e-6f);
			const static float3 betaM = float3(21e-6f, 21e-6f, 21e-6f);

			float _ScaleHeightR;
			float _ScaleHeightM;
			float3 _CameraHeight;

			float _IsToneMap;
			float _Exposure;
			float _AtmosphereThickness;
			float _MeanCosine;
			#define _MeanCosine2 = _MeanCosine * _MeanCosine;
			int _ViewDirSampleNum;
			int _LightDirSampleNum;

			sampler2D _MainTex;

			struct Input {
				float2 uv_MainTex;
			};

			struct VSInput
			{
				float4 pos : POSITION;
				float2 uv: TEXCOORD0;
			};

			struct VSOutput
			{
				float4 pos: SV_POSITION;
				float2 uv: TEXCOORD0;
				float3 worldPos: TEXCOORD1;
				float3 vertex: TEXCOORD2;
			};

			VSOutput vert(VSInput input)
			{
				VSOutput output;
				output.pos = UnityObjectToClipPos(input.pos);
				output.worldPos = mul(UNITY_MATRIX_M, input.pos);
				output.uv = input.uv;
				output.vertex = input.pos;
				return output;
			}
			bool solveQuadratic(float a, float b, float c, inout float x1, inout float x2)
			{
				if (b == 0) {
					// Handle special case where the the two vector ray.dir and V are perpendicular
					// with V = ray.orig - sphere.centre
					if (a == 0) return false;
					x1 = 0; x2 = sqrt(-c / a);
					return true;
				}
				float discr = b * b - 4 * a * c;

				if (discr < 0) return false;

				float q = (b < 0.f) ? -0.5f * (b - sqrt(discr)) : -0.5f * (b + sqrt(discr));
				x1 = q / a;
				x2 = c / q;

				return true;
			}

			bool raySphereIntersect(const float3 orig, const float3 dir, float radius, inout float t0, inout float t1)
			{
				// They ray dir is normalized so A = 1 
				float A = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
				float B = 2 * (dir.x * orig.x + dir.y * orig.y + dir.z * orig.z);
				float C = orig.x * orig.x + orig.y * orig.y + orig.z * orig.z - radius * radius;

				if (!solveQuadratic(A, B, C, t0, t1)) return false;

				if (t0 > t1)
				{
					float temp = t0;
					t0 = t1;
					t1 = temp;
				}

				return true;
			}

			float length(float3 val)
			{
				return sqrt(val.x * val.x + val.y * val.y + val.z * val.z);
			}

			float3 computeIncidentLight(float3 orig, float3 dir, float tmin, float tmax)
			{
				float t0, t1;
				if (!raySphereIntersect(orig, dir, AtmosphereRadius, t0, t1) || t1 < 0) return 0;
				if (t0 > tmin && t0 > 0) tmin = t0;
				if (t1 < tmax) tmax = t1;
				float segmentLength = (tmax - tmin) / _ViewDirSampleNum;
				float tCurrent = tmin;
				float3 sumR = 0;
				float3 sumM = 0;
				float opticalDepthR = 0;
				float opticalDepthM = 0;
				float3 sunDirection = normalize(_WorldSpaceLightPos0.xyz);
				float mu = dot(dir, sunDirection); // mu in the paper which is the cosine of the angle between the sun direction and the ray direction
				float g = _MeanCosine;
				float phaseR = 3.f / (16.f * M_PI) * (1 + mu * mu);
				float phaseM = 3.f / (8.f * M_PI) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));

				for (int i = 0; i < _ViewDirSampleNum; ++i) {
					float3 samplePosition = orig + (tCurrent + segmentLength * 0.5f) * dir;
					float height = length(samplePosition) - EarthRadius;
					// compute optical depth for light
					float hr = exp(-height / _ScaleHeightR) * segmentLength;
					float hm = exp(-height / _ScaleHeightM) * segmentLength;
					opticalDepthR += hr;
					opticalDepthM += hm;
					// light optical depth
					float t0Light, t1Light;
					raySphereIntersect(samplePosition, sunDirection, AtmosphereRadius, t0Light, t1Light);
					float segmentLengthLight = t1Light / _LightDirSampleNum;
					float tCurrentLight = 0;
					float opticalDepthLightR = 0;
					float opticalDepthLightM = 0;
					for (int j = 0; j < _LightDirSampleNum; ++j) {
						float3 samplePositionLight = samplePosition + (tCurrentLight + segmentLengthLight * 0.5f) * sunDirection;
						float heightLight = length(samplePositionLight) - EarthRadius;
						if (heightLight < 0) break;
						opticalDepthLightR += exp(-heightLight / _ScaleHeightR) * segmentLengthLight;
						opticalDepthLightM += exp(-heightLight / _ScaleHeightM) * segmentLengthLight;
						tCurrentLight += segmentLengthLight;
					}
					if (j == _LightDirSampleNum) {
						float3 tau = betaR * (opticalDepthR + opticalDepthLightR) + betaM * 1.1f * (opticalDepthM + opticalDepthLightM);
						float3 attenuation = float3(exp(-tau));
						sumR += attenuation * hr;
						sumM += attenuation * hm;
					}
					tCurrent += segmentLength;
				}
				return (sumR * betaR * phaseR + sumM * betaM * phaseM) * _Exposure;
			}

			// Uncharted 2 tonemap from http://filmicgames.com/archives/75
			inline float3 FilmicTonemap(float3 x)
			{
				// Consts need to be defined locally due to Unity bug
				// global consts result in black screen for some reason

				const float A = 0.15;
				const float B = 0.50;
				const float C = 0.10;
				const float D = 0.20;
				const float E = 0.02;
				const float F = 0.30;

				return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E / F;
			}

			float3 ApplyToneMap(float3 color)
			{
				const float W = 11.2;
				color *= _ToneMapExposure;
				float ExposureBias = 2.0f;
				float3 curr = FilmicTonemap(ExposureBias * color);
				float3 whiteScale = 1.0f / FilmicTonemap(W);
				color = curr * whiteScale;
				return color;
			}

			inline float3 UVToCameraRay(float2 uv)
			{
				float4 cameraRay = float4(uv * 2.0 - 1.0, 1.0, 1.0);
				cameraRay = mul(_InverseProjection, cameraRay);
				cameraRay = cameraRay / cameraRay.w;

				//return mul(_InverseView, cameraRay.xyz);
				return mul((float3x3)_InverseView, cameraRay.xyz);
			}

			/*
			The functions to compute shadows and light shafts must be defined before we
			can use them in the main shader function, so we define them first. Testing if
			a point is in the shadow of the sphere S is equivalent to test if the
			corresponding light ray intersects the sphere, which is very simple to do.
			However, this is only valid for a punctual light source, which is not the case
			of the Sun. In the following function we compute an approximate (and biased)
			soft shadow by taking the angular size of the Sun into account:
			*/
			float GetSunVisibility(float3 _point, float3 sun_direction)
			{
				float3 p = _point - kSphereCenter;
				float p_dot_v = dot(p, sun_direction);
				float p_dot_p = dot(p, p);
				float ray_sphere_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
				float distance_to_intersection = -p_dot_v - sqrt(max(0.0, kSphereRadius * kSphereRadius - ray_sphere_center_squared_distance));

				if (distance_to_intersection > 0.0)
				{
					// Compute the distance between the view ray and the sphere, and the
					// corresponding (tangent of the) subtended angle. Finally, use this to
					// compute an approximate sun visibility.
					float ray_sphere_distance = kSphereRadius - sqrt(ray_sphere_center_squared_distance);
					float ray_sphere_angular_distance = -ray_sphere_distance / p_dot_v;

					return smoothstep(1.0, 0.0, ray_sphere_angular_distance / sun_size.x);
				}

				return 1.0;
			}

			/*
			The sphere also partially occludes the sky light, and we approximate this
			effect with an ambient occlusion factor. The ambient occlusion factor due to a
			sphere is given in <a href=
			"http://webserver.dmt.upm.es/~isidoro/tc3/Radiation%20View%20factors.pdf"
			>Radiation View Factors</a> (Isidoro Martinez, 1995). In the simple case where
			the sphere is fully visible, it is given by the following function:
			*/
			float GetSkyVisibility(float3 _point)
			{
				float3 p = _point - kSphereCenter;
				float p_dot_p = dot(p, p);
				return 1.0 + p.y / sqrt(p_dot_p) * kSphereRadius * kSphereRadius / p_dot_p;
			}

			/*
			To compute light shafts we need the intersections of the view ray with the
			shadow volume of the sphere S. Since the Sun is not a punctual light source this
			shadow volume is not a cylinder but a cone (for the umbra, plus another cone for
			the penumbra, but we ignore it here):
			*/
			void GetSphereShadowInOut(float3 view_direction, float3 sun_direction, out float d_in, out float d_out)
			{
				float3 camera = _WorldSpaceCameraPos;
				float3 pos = camera - kSphereCenter;
				float pos_dot_sun = dot(pos, sun_direction);
				float view_dot_sun = dot(view_direction, sun_direction);
				float k = sun_size.x;
				float l = 1.0 + k * k;
				float a = 1.0 - l * view_dot_sun * view_dot_sun;
				float b = dot(pos, view_direction) - l * pos_dot_sun * view_dot_sun -
					k * kSphereRadius * view_dot_sun;
				float c = dot(pos, pos) - l * pos_dot_sun * pos_dot_sun -
					2.0 * k * kSphereRadius * pos_dot_sun - kSphereRadius * kSphereRadius;
				float discriminant = b * b - a * c;
				if (discriminant > 0.0)
				{
					d_in = max(0.0, (-b - sqrt(discriminant)) / a);
					d_out = (-b + sqrt(discriminant)) / a;
					// The values of d for which delta is equal to 0 and kSphereRadius / k.
					float d_base = -pos_dot_sun / view_dot_sun;
					float d_apex = -(pos_dot_sun + kSphereRadius / k) / view_dot_sun;

					if (view_dot_sun > 0.0)
					{
						d_in = max(d_in, d_apex);
						d_out = a > 0.0 ? min(d_out, d_base) : d_base;
					}
					else
					{
						d_in = a > 0.0 ? max(d_in, d_base) : d_base;
						d_out = min(d_out, d_apex);
					}
				}
				else
				{
					d_in = 0.0;
					d_out = 0.0;
				}
			}

			RadianceSpectrum GetSolarRadiance(AtmosphereParameters atmosphere)
			{
				return atmosphere.solar_irradiance / (PI * sun_angular_radius * sun_angular_radius);
			}

			RadianceSpectrum GetSkyRadiance(AtmosphereParameters atmosphere,
				Position camera, Direction view_ray, Length shadow_length,
				Direction sun_direction, out DimensionlessSpectrum transmittance)
			{
				return GetSkyRadiance(atmosphere, transmittance_texture,
					scattering_texture, single_mie_scattering_texture,
					camera, view_ray, shadow_length, sun_direction, transmittance);
			}

			RadianceSpectrum GetSkyRadianceToPoint(AtmosphereParameters atmosphere,
				Position camera, Position _point, Length shadow_length,
				Direction sun_direction, out DimensionlessSpectrum transmittance)
			{
				return GetSkyRadianceToPoint(atmosphere, transmittance_texture,
					scattering_texture, single_mie_scattering_texture,
					camera, _point, shadow_length, sun_direction, transmittance);
			}

			IrradianceSpectrum GetSunAndSkyIrradiance(AtmosphereParameters atmosphere,
				Position p, Direction normal, Direction sun_direction,
				out IrradianceSpectrum sky_irradiance)
			{
				return GetSunAndSkyIrradiance(atmosphere, transmittance_texture,
					irradiance_texture, p, normal, sun_direction, sky_irradiance);
			}

#if 0
			float4 frag(VSOutput input) : SV_Target0
			{
				//return tex2D(_TransmittanceTex, input.uv);
				//return tex3D(_ScatteringTex, float3(input.uv, 0));
				AtmosphereParameters atmosphere = InitAtmosphereParameters();
			//float3 V = normalize(UVToCameraRay(input.uv));
			float3 V = normalize(input.worldPos);
				float3 L = normalize(_WorldSpaceLightPos0.xyz);
				float earthRadius = 6360;
				float atmosphereRadius = 6420;
				float altitude = earthRadius + _CameraHeight.y;
				float3 origin = float3(0, earthRadius, 0) + _CameraHeight;
				float x0, x1 = kInf;
				float y0, y1 = kInf;
				bool isIntersectGround = raySphereIntersect(origin, V, earthRadius, x0, x1);
				bool isIntersectAtmosphere = raySphereIntersect(origin, V, atmosphereRadius, y0, y1);

				float mu = dot(V, float3(0, 1, 0));// / altitude * y1;
				float mu_s = dot(L, float3(0, 1, 0));// / altitude;
				float nu = dot(L, V);// / y1;

				float3 scattering = GetScattering(atmosphere, _ScatteringTex, _MieScatteringTex, _ScatteringTex, altitude, mu, mu_s, nu, isIntersectGround, 1);
				float3 output = ApplyToneMap(scattering);
				return float4(output * _Exposure, 1);
				//float4 color = 0;

				//float3 origin = float3(0, EarthRadius, 0) + _CameraHeight;
				////float3 origin = _WorldSpaceCameraPos.xyz + EarthRadius + 1000;
				//float3 rayDir = normalize(input.worldPos);
				////float3 rayDir = normalize(input.worldPos * AtmosphereRadius - origin.xyz);

				//// TODO single scattering texture + mie_scattering textureで単一散乱実装

				//float x0, x1, xMax = kInf;
				//if (!raySphereIntersect(origin, rayDir, EarthRadius, x0, x1) && x1 > 0)
				//{
				//	// 衝突したら近い点を衝突点とする
				//	xMax = max(0, x0);
				//	//return float4(1, 0, 0, 1);
				//}

				//color.rgb += computeIncidentLight(origin, rayDir, 0, xMax);

				//if(_IsToneMap)
				//{
				//	color.r = color.r < 1.413f ? pow(color.r * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-color.r);
				//	color.g = color.g < 1.413f ? pow(color.g * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-color.g);
				//	color.b = color.b < 1.413f ? pow(color.b * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-color.b);
				//}
				//color.a = 1;
				//return color;

//#if SCATTER_MODE == SCATTER_MODE_RAYLEIGH
//				return 1;
//#elif SCATTER_MODE == SCATTER_MODE_MIE
//				return 0.5;
//#else
//				return 0;
//#endif
//				return 1;
			}

#else
			fixed4 frag(VSOutput i) : SV_Target
			{
				//return float4(1, 0, 0, 1);
			//return tex3D(delta_scattering_density_texture, float3(i.uv, depth_test));
			//return tex2D(transmittance_texture, i.uv);
			//return tex2D(irradiance_texture, i.uv);
			//return tex3D(scattering_texture, float3(i.uv, depth_test));
			//return tex3D(single_mie_scattering_texture, float3(i.uv, depth_test));
			AtmosphereParameters atmosphere = InitAtmosphereParameters();

			float3 camera = _WorldSpaceCameraPos;
			float3 view_direction = normalize(i.worldPos);
			//float3 view_direction = normalize(i.view_ray);

			float shadow_in = 0;
			float shadow_out = 0;
			//GetSphereShadowInOut(view_direction, sun_direction, shadow_in, shadow_out);

			// Hack to fade out light shafts when the Sun is very close to the horizon.
			float lightshaft_fadein_hack = smoothstep(
				0.02, 0.04, dot(normalize(camera - earth_center), sun_direction));

			// Compute the distance between the view ray line and the Earth center,
			// and the distance between the camera and the intersection of the view
			// ray with the ground (or NaN if there is no intersection).
			float3 p = camera - earth_center;
			float p_dot_v = dot(p, view_direction);
			float p_dot_p = dot(p, p);
			float ray_earth_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
			float distance_to_intersection = -p_dot_v - sqrt(earth_center.y * earth_center.y - ray_earth_center_squared_distance);
			// Compute the radiance reflected by the ground, if the ray intersects it.
			float ground_alpha = 0.0;
			float3 ground_radiance = float3(0, 0, 0);
			//if (distance_to_intersection > 0.0)
			//{
			//	float3 _point = camera + view_direction * distance_to_intersection;
			//	float3 normal = normalize(_point - earth_center);

			//	// Compute the radiance reflected by the ground.
			//	float3 sky_irradiance;
			//	float3 sun_irradiance = GetSunAndSkyIrradiance(atmosphere, _point - earth_center, normal, sun_direction, sky_irradiance);

			//	float sunVis = GetSunVisibility(_point, sun_direction);
			//	float skyVis = GetSkyVisibility(_point);

			//	ground_radiance = kGroundAlbedo * (1.0 / PI) * (sun_irradiance * sunVis + sky_irradiance * skyVis);

			//	float shadow_length = 0;// max(0.0, min(shadow_out, distance_to_intersection) - shadow_in) * lightshaft_fadein_hack;

			//	float3 transmittance;
			//	float3 in_scatter = GetSkyRadianceToPoint(atmosphere, camera - earth_center, _point - earth_center, shadow_length, sun_direction, transmittance);

			//	ground_radiance = ground_radiance * transmittance + in_scatter;
			//	ground_alpha = 1.0;
			//}

			/*
			Finally, we compute the radiance and transmittance of the sky, and composite
			together, from back to front, the radiance and opacities of all the objects of
			the scene:
			*/

			// Compute the radiance of the sky.
			float shadow_length = 0;// max(0.0, shadow_out - shadow_in) * lightshaft_fadein_hack;
			float3 transmittance;
			float3 radiance = GetSkyRadiance(atmosphere, camera - earth_center, view_direction, shadow_length, sun_direction, transmittance);

			// If the view ray intersects the Sun, add the Sun radiance.
			if (dot(view_direction, sun_direction) > sun_size.y)
			{
				radiance = radiance + transmittance * GetSolarRadiance(atmosphere);
			}

			radiance = lerp(radiance, ground_radiance, ground_alpha);

			radiance = pow(float3(1,1,1) - exp(-radiance / white_point * _Exposure), 1.0 / 2.2);

			return float4(radiance, 1);
			}
#endif

			ENDCG
		}
	}
	FallBack Off
}
