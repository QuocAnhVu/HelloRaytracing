use glam::Vec3;
use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{random_unit_vec, Color, Hit, Ray, WHITE};

#[derive(Copy, Clone)]
pub enum Material {
    Lambertian(Lambertian),
    Metallic(Metallic),
    Dielectric(Dielectric),
}

impl Material {
    pub fn lambertian(albedo: Color) -> Self {
        Material::Lambertian(Lambertian { albedo })
    }
    pub fn metallic(albedo: Color, fuzz: f32) -> Self {
        Material::Metallic(Metallic { albedo, fuzz })
    }
    pub fn dielectric(refraction_index: f32) -> Self {
        Material::Dielectric(Dielectric { refraction_index })
    }

    pub fn scatter(&self, ray: &Ray, hit: &Hit) -> (Color, Ray) {
        match self {
            Material::Lambertian(mat) => mat.scatter(ray, hit),
            Material::Metallic(mat) => mat.scatter(ray, hit),
            Material::Dielectric(mat) => mat.scatter(ray, hit),
        }
    }
}

trait Scattering {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> (Color, Ray);
}

#[derive(Copy, Clone)]
struct Lambertian {
    pub albedo: Vec3,
}
impl Scattering for Lambertian {
    fn scatter(&self, _ray: &Ray, hit: &Hit) -> (Color, Ray) {
        let mut scatter_direction = hit.normal + random_unit_vec();
        if scatter_direction
            .abs()
            .cmplt(Vec3::splat(f32::EPSILON))
            .all()
        {
            scatter_direction = hit.normal;
        }
        (
            self.albedo,
            Ray {
                transl: hit.transl,
                direction: scatter_direction,
            },
        )
    }
}

#[derive(Copy, Clone)]
struct Metallic {
    albedo: Vec3,
    fuzz: f32,
}
impl Scattering for Metallic {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> (Color, Ray) {
        let reflected_direction = reflect(ray.direction, hit.normal);
        let reflected_direction = reflected_direction.normalize() + self.fuzz * random_unit_vec();
        (
            self.albedo,
            Ray {
                transl: hit.transl,
                direction: reflected_direction,
            },
        )
    }
}

#[derive(Copy, Clone)]
struct Dielectric {
    refraction_index: f32,
}
impl Scattering for Dielectric {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> (Color, Ray) {
        let refraction_index = if hit.is_front_face {
            self.refraction_index.recip()
        } else {
            self.refraction_index
        };

        let direction = ray.direction.normalize();
        let cos_theta = (-direction).dot(hit.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let must_reflect = refraction_index * sin_theta > 1.0;
        let mut rng = thread_rng();
        let proportion = Uniform::new(0.0, 1.0);
        let refracted_direction =
            if must_reflect || reflectance(cos_theta, refraction_index) > rng.sample(proportion) {
                reflect(direction, hit.normal)
            } else {
                refract(direction, hit.normal, refraction_index)
            };

        (
            WHITE,
            Ray {
                transl: hit.transl,
                direction: refracted_direction,
            },
        )
    }
}

fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    incident - 2.0 * incident.dot(normal) * normal
}

fn refract(incident: Vec3, normal: Vec3, refraction_index: f32) -> Vec3 {
    let cos_theta = (-incident).dot(normal).min(1.0);

    let ray_perpendicular = refraction_index * (incident + cos_theta * normal);
    let ray_parallel = (1.0 - ray_perpendicular.length_squared()).abs().sqrt() * -normal;
    ray_perpendicular + ray_parallel
}

// Schlick's approximation for reflectance
fn reflectance(cosine: f32, refraction_index: f32) -> f32 {
    let r0 = ((1.0 - refraction_index) / (1.0 + refraction_index)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}
