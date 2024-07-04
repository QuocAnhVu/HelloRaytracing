use std::{cmp, f32::consts::TAU, fs::File, io, rc::Rc};

use glam::Vec3;
use indicatif::ProgressBar;
use itertools::{self, Itertools};
use rand::{distributions::Uniform, thread_rng, Rng};

type Color = Vec3;
const WHITE: Color = Color::new(1.0, 1.0, 1.0);
const BLACK: Color = Color::new(0.0, 0.0, 0.0);

fn main() -> io::Result<()> {
    // Define scene
    let mut scene = Scene::new();
    scene.add(Sphere {
        transl: Vec3::new(0.0, -100.5, -1.0),
        radius: 100.0,
        mat: Rc::new(Lambertian {
            albedo: Color::new(0.8, 0.8, 0.0),
        }),
    });
    scene.add(Sphere {
        transl: Vec3::new(0.0, 0.0, -1.2),
        radius: 0.5,
        mat: Rc::new(Lambertian {
            albedo: Color::new(0.1, 0.2, 0.5),
        }),
    });
    scene.add(Sphere {
        transl: Vec3::new(-1.0, 0.0, -1.0),
        radius: 0.5,
        mat: Rc::new(Dielectric {
            refraction_index: 1.0 / 1.33,
        }),
        // mat: Rc::new(Metallic {
        //     albedo: Color::new(0.8, 0.8, 0.8),
        //     fuzz: 0.3,
        // }),
    });
    scene.add(Sphere {
        transl: Vec3::new(1.0, 0.0, -1.0),
        radius: 0.5,
        mat: Rc::new(Metallic {
            albedo: Color::new(0.8, 0.6, 0.2),
            fuzz: 1.0,
        }),
    });

    // Define camera
    let camera = Camera::new(
        Vec3::new(-2.0, 2.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 1.0, 0.0),
    );

    let pixels = camera.pixels();
    let bar = ProgressBar::new(
        (Camera::IMG_HEIGHT * Camera::IMG_WIDTH * Camera::SAMPLES_PER_PIXEL) as u64,
    );
    let mut colors: Vec<(Pixel, Color)> = pixels
        .map(|pixel| (pixel, camera.render(&scene, pixel)))
        .map(|x| {
            bar.inc(Camera::SAMPLES_PER_PIXEL as u64);
            x
        })
        // .par_bridge()
        .collect();
    colors.sort_unstable_by_key(|(Pixel(i, j), _)| (*j, *i));

    let mut file = File::create("image.ppm")?;
    write_header(Camera::IMG_WIDTH, Camera::IMG_HEIGHT, &mut file);
    colors
        .iter()
        .map(|(_, color)| color)
        .for_each(move |color| {
            write_pixel(color, &mut file);
        });
    Ok(())
}

fn write_header(image_width: u32, image_height: u32, mut writer: impl io::Write) {
    let header = format!(
        "P3\n\
         {image_width} {image_height}\n\
         255\n"
    );
    writer
        .write_all(header.as_bytes())
        .expect("Failed to write header!");
    writer.flush().unwrap();
}
fn write_pixel(pixel: &Color, mut writer: impl io::Write) {
    const MAX_COLOR: f32 = 256.0 - f32::EPSILON;
    let mut r = pixel.x.clamp(0.0, 1.0);
    let mut g = pixel.y.clamp(0.0, 1.0);
    let mut b = pixel.z.clamp(0.0, 1.0);
    r = gamma_correction(r);
    g = gamma_correction(g);
    b = gamma_correction(b);
    let r_byte = (MAX_COLOR * r) as u8;
    let g_byte = (MAX_COLOR * g) as u8;
    let b_byte = (MAX_COLOR * b) as u8;

    let line = format!("{r_byte} {g_byte} {b_byte}\n");
    writer
        .write_all(line.as_bytes())
        .expect("Failed to write line: {line}");
    writer.flush().unwrap();
}
fn gamma_correction(linear_brightness: f32) -> f32 {
    if linear_brightness > 0.0 {
        linear_brightness.sqrt()
    } else {
        0.0
    }
}

#[derive(Copy, Clone)]
struct Pixel(u32, u32);
struct Camera {
    camera_center: Vec3,
    pixel_origin: Vec3,
    pixel_du: Vec3,
    pixel_dv: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
}
impl Camera {
    const IMG_HEIGHT: u32 = 256;
    const IMG_WIDTH: u32 = 1024;
    const SAMPLES_PER_PIXEL: u32 = 8;
    const VERTICAL_FOV: f32 = TAU / 8.0;

    fn new(look_from: Vec3, look_at: Vec3, up: Vec3) -> Self {
        // Camera
        let focal_length = (look_from - look_at).length();
        let camera_center = look_from;
        let h = (Self::VERTICAL_FOV / 2.0).tan();
        let viewport_height = 2.0_f32 * h * focal_length;
        let viewport_width = viewport_height * (Self::IMG_WIDTH as f32 / Self::IMG_HEIGHT as f32);

        // Calculate u, v, w unit vectors for camera coordinate frame
        let w = (look_from - look_at).normalize();
        let u = up.cross(w).normalize();
        let v = w.cross(u);

        // Calculate vectors across horizontal/down vertical viewport edges
        let viewport_u = viewport_width * u;
        let viewport_v = viewport_height * -v;

        // Calculate horizontal/vertical delta vectors from pixel to pixel
        let pixel_du = viewport_u / Self::IMG_WIDTH as f32;
        let pixel_dv = viewport_v / Self::IMG_HEIGHT as f32;

        // Calculate the location of the upper left pixel
        let viewport_origin = camera_center - focal_length * w - (viewport_u + viewport_v) / 2.0;
        let pixel_origin = viewport_origin + 0.5 * (pixel_du + pixel_dv);

        Camera {
            camera_center,
            pixel_origin,
            pixel_du,
            pixel_dv,
            u,
            v,
            w,
        }
    }

    fn pixels(&self) -> impl Iterator<Item = Pixel> {
        (0..Self::IMG_HEIGHT)
            .cartesian_product(0..Self::IMG_WIDTH)
            .map(|(j, i)| Pixel(i, j))
    }
    fn render(&self, scene: &Scene, pixel: Pixel) -> Color {
        const MAX_DEPTH: u32 = 64;

        let Pixel(i, j) = pixel;
        let mut rng = thread_rng();
        let offset = Uniform::new(-0.5, 0.5);
        let mut pixel_color = BLACK;
        for _ in 0..Self::SAMPLES_PER_PIXEL {
            let pixel_center = self.pixel_origin
                + ((i as f32 + rng.sample(offset)) * self.pixel_du)
                + ((j as f32 + rng.sample(offset)) * self.pixel_dv);
            let ray = Ray {
                transl: self.camera_center,
                direction: pixel_center - self.camera_center,
            };
            pixel_color += scene.ray_color(&ray, MAX_DEPTH) / Self::SAMPLES_PER_PIXEL as f32;
        }
        pixel_color
    }
}

struct Ray {
    transl: Vec3,
    direction: Vec3,
}
impl Ray {
    fn at(&self, distance: f32) -> Vec3 {
        self.transl + distance * self.direction
    }
}

struct Scene {
    objects: Vec<Box<dyn Hittable>>,
}
impl Scene {
    fn new() -> Self {
        Scene {
            objects: Vec::new(),
        }
    }
    fn add(&mut self, object: impl Hittable + 'static) {
        self.objects.push(Box::new(object));
    }
    fn ray_color(&self, ray: &Ray, depth: u32) -> Color {
        if depth == 0 {
            return BLACK;
        }
        let mut closest_hit: Option<Hit> = None;
        for object in &self.objects {
            let candidate_hit = object.hit(ray, f32::EPSILON, f32::INFINITY);
            closest_hit = match (closest_hit, candidate_hit) {
                (None, None) => None,
                (a @ Some(_), None) => a,
                (None, b @ Some(_)) => b,
                (Some(a), Some(b)) => {
                    if a.t < b.t {
                        Some(a)
                    } else {
                        Some(b)
                    }
                }
            };
        }
        match closest_hit {
            Some(hit) => {
                let (attenuation, scattered) = hit.mat.scatter(ray, &hit);
                attenuation * self.ray_color(&scattered, depth - 1)
            }
            None => {
                let direction = ray.direction.normalize();
                let alpha = 0.5 * (direction.y + 1.0);
                (1.0 - alpha) * WHITE + alpha * Color::new(0.5, 0.7, 1.0)
            }
        }
    }
}
fn random_unit_vec() -> Vec3 {
    let mut rng = thread_rng();
    let dimensions = Uniform::new(-1.0, 1.0);
    Vec3::new(
        rng.sample(dimensions),
        rng.sample(dimensions),
        rng.sample(dimensions),
    )
    .normalize()
}

struct Hit {
    t: f32,
    transl: Vec3,
    normal: Vec3,
    is_front_face: bool,
    mat: Rc<dyn Material>,
}
trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<Hit>;
}

struct Sphere {
    transl: Vec3,
    radius: f32,
    mat: Rc<dyn Material>,
}
impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<Hit> {
        let oc = self.transl - ray.transl;
        let a = ray.direction.length_squared();
        let h = ray.direction.dot(oc);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            return None;
        }

        let mut root = (h - discriminant.sqrt()) / a;
        if !(t_min..=t_max).contains(&root) {
            root = (h + discriminant.sqrt()) / a;
            if !(t_min..=t_max).contains(&root) {
                return None;
            }
        }

        let t = root;
        let transl = ray.at(t);
        let mut normal = (transl - self.transl) / self.radius;
        let is_front_face = normal.dot(ray.direction) < 0.0;
        if !is_front_face {
            normal = -normal;
        }
        Some(Hit {
            t,
            transl,
            normal,
            is_front_face,
            mat: self.mat.clone(),
        })
    }
}

trait Material {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> (Color, Ray);
}

struct Lambertian {
    pub albedo: Vec3,
}
impl Material for Lambertian {
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

struct Metallic {
    albedo: Vec3,
    fuzz: f32,
}
impl Material for Metallic {
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

struct Dielectric {
    refraction_index: f32,
}
impl Material for Dielectric {
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
