mod material;

use std::{f32::consts::TAU, fs::File, io};

use glam::Vec3;
use indicatif::ProgressBar;
use itertools::{self, Itertools};
use material::Material;
use rand::{distributions::Uniform, thread_rng, Rng};
use rayon::prelude::*;

type Color = Vec3;
const WHITE: Color = Color::new(1.0, 1.0, 1.0);
const BLACK: Color = Color::new(0.0, 0.0, 0.0);

fn main() -> io::Result<()> {
    // Define scene
    let mut scene = Scene::new();
    scene.add(Sphere {
        transl: Vec3::new(0.0, -100.5, -1.0),
        radius: 100.0,
        mat: Material::lambertian(Color::new(0.8, 0.8, 0.0)),
    });
    scene.add(Sphere {
        transl: Vec3::new(0.0, 0.0, -1.2),
        radius: 0.5,
        mat: Material::lambertian(Color::new(0.1, 0.2, 0.5)),
    });
    scene.add(Sphere {
        transl: Vec3::new(-1.0, 0.0, -1.0),
        radius: 0.5,
        mat: Material::dielectric(1.0 / 1.33),
    });
    scene.add(Sphere {
        transl: Vec3::new(1.0, 0.0, -1.0),
        radius: 0.5,
        mat: Material::metallic(Color::new(0.8, 0.6, 0.2), 1.0),
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
        .par_bridge()
        .map(|pixel| (pixel, camera.render(&scene, pixel)))
        .map(|x| {
            bar.inc(Camera::SAMPLES_PER_PIXEL as u64);
            x
        })
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
}
impl Camera {
    const IMG_HEIGHT: u32 = 1080;
    const IMG_WIDTH: u32 = 1920;
    const SAMPLES_PER_PIXEL: u32 = 256;
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
        let offset = Uniform::new(-0.5, 0.5);
        let pixel_color = (0..Self::SAMPLES_PER_PIXEL)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let pixel_center = self.pixel_origin
                    + ((i as f32 + rng.sample(offset)) * self.pixel_du)
                    + ((j as f32 + rng.sample(offset)) * self.pixel_dv);
                let ray = Ray {
                    transl: self.camera_center,
                    direction: pixel_center - self.camera_center,
                };
                scene.ray_color(&ray, MAX_DEPTH)
            })
            .sum::<Color>();
        pixel_color / (Self::SAMPLES_PER_PIXEL as f32)
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
    objects: Vec<Box<dyn Hittable + Send + Sync>>,
}
impl Scene {
    fn new() -> Self {
        Scene {
            objects: Vec::new(),
        }
    }
    fn add(&mut self, object: impl Hittable + Send + Sync + 'static) {
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
    mat: Material,
}
trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<Hit>;
}

struct Sphere {
    transl: Vec3,
    radius: f32,
    mat: Material,
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
            mat: self.mat,
        })
    }
}
