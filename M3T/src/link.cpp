// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber, German Aerospace Center (DLR)

#include <m3t/link.h>
#include <m3t/texture_modality.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace m3t {

Link::Link(const std::string &name, const std::shared_ptr<Body> &body_ptr,
           const Transform3fA &body2joint_pose,
           const Transform3fA &joint2parent_pose,
           const Transform3fA &link2world_pose,
           const std::array<bool, 6> &free_directions,
           bool fixed_body2joint_pose)
    : name_{name},
      body_ptr_{body_ptr},
      body2joint_pose_{body2joint_pose},
      default_body2joint_pose_{body2joint_pose},
      joint2parent_pose_{joint2parent_pose},
      default_joint2parent_pose_{joint2parent_pose},
      link2world_pose_{link2world_pose},
      free_directions_{free_directions},
      fixed_body2joint_pose_{fixed_body2joint_pose} {}

Link::Link(const std::string &name, const std::filesystem::path &metafile_path,
           const std::shared_ptr<Body> &body_ptr)
    : name_{name}, metafile_path_{metafile_path}, body_ptr_{body_ptr} {}

bool Link::SetUp() {
  set_up_ = false;
  jacobian_defined_ = false;
  if (!metafile_path_.empty())
    if (!LoadMetaData()) return false;

  if (body_ptr_) {
    // Check if all required objects are set up
    if (!body_ptr_->set_up()) {
      std::cerr << "Body " << body_ptr_->name() << " was not set up"
                << std::endl;
      return false;
    }
    for (auto &modality_ptr : modality_ptrs_) {
      if (!modality_ptr->set_up()) {
        std::cerr << "Modality " << modality_ptr->name() << " was not set up"
                  << std::endl;
        return false;
      }
    }

    // Check if all modalities reference the main body
    for (auto &modality_ptr : modality_ptrs_) {
      if (modality_ptr->body_ptr()->name() != body_ptr_->name()) {
        std::cerr << "Modalities that reference different bodies than the main "
                  << "body were assigned to link " << name_ << std::endl;
        return false;
      }
    }
  } else {
    if (!modality_ptrs_.empty()) {
      std::cerr << "No main body assigned to link " << name_ << std::endl;
      return false;
    }
  }

  set_up_ = true;
  return true;
}

bool Link::AddModality(const std::shared_ptr<Modality> &modality_ptr) {
  set_up_ = false;
  if (!AddPtrIfNameNotExists(modality_ptr, &modality_ptrs_)) {
    std::cerr << "Modality " << modality_ptr->name() << " already exists"
              << std::endl;
    return false;
  }
  return true;
}

bool Link::DeleteModality(const std::string &name) {
  set_up_ = false;
  if (!DeletePtrIfNameExists(name, &modality_ptrs_)) {
    std::cerr << "Modality " << name << " not found" << std::endl;
    return false;
  }
  return true;
}

void Link::ClearModalities() {
  set_up_ = false;
  modality_ptrs_.clear();
}

bool Link::AddChildLink(const std::shared_ptr<Link> &child_link_ptr) {
  set_up_ = false;
  if (!AddPtrIfNameNotExists(child_link_ptr, &child_link_ptrs_)) {
    std::cerr << "Child link " << child_link_ptr->name() << " already exists"
              << std::endl;
    return false;
  }
  return true;
}

bool Link::DeleteChildLink(const std::string &name) {
  set_up_ = false;
  if (!DeletePtrIfNameExists(name, &child_link_ptrs_)) {
    std::cerr << "Child link " << name << " not found" << std::endl;
    return false;
  }
  return true;
}

void Link::ClearChildLinks() {
  set_up_ = false;
  child_link_ptrs_.clear();
}

void Link::set_name(const std::string &name) { name_ = name; }

void Link::set_metafile_path(const std::filesystem::path &metafile_path) {
  metafile_path_ = metafile_path;
  set_up_ = false;
}

void Link::set_body_ptr(const std::shared_ptr<Body> &body_ptr) {
  body_ptr_ = body_ptr;
  set_up_ = false;
}

void Link::set_body2joint_pose(const Transform3fA &body2joint_pose) {
  body2joint_pose_ = body2joint_pose;
  default_body2joint_pose_ = body2joint_pose;
}

void Link::set_joint2parent_pose(const Transform3fA &joint2parent_pose) {
  joint2parent_pose_ = joint2parent_pose;
  default_joint2parent_pose_ = joint2parent_pose;
}

void Link::set_link2world_pose(const Transform3fA &link2world_pose) {
  link2world_pose_ = link2world_pose;
}

void Link::set_free_directions(const std::array<bool, 6> &free_directions) {
  free_directions_ = free_directions;
  set_up_ = false;
}

void Link::set_fixed_body2joint_pose(bool fixed_body2joint_pose) {
  fixed_body2joint_pose_ = fixed_body2joint_pose;
}

void Link::set_use_reliability_weighted_fusion(
    bool use_reliability_weighted_fusion) {
  use_reliability_weighted_fusion_ = use_reliability_weighted_fusion;
}

void Link::SetDirectionalTemporalPrior(const Transform3fA &predicted_pose) {
  predicted_pose_ = predicted_pose;
  directional_temporal_prior_valid_ = true;
}

void Link::ClearDirectionalTemporalPrior() {
  directional_temporal_prior_valid_ = false;
}

bool Link::DefineJacobian(int jacobian_size, int first_jacobian_index) {
  if (!IsSetup(false)) return false;
  jacobian_.resize(Eigen::NoChange, jacobian_size);
  first_jacobian_index_ = first_jacobian_index;
  jacobian_defined_ = true;
  return true;
}

bool Link::CalculateJacobian(const std::shared_ptr<Link> &parent_link_ptr) {
  if (!IsSetup(true)) return false;

  // project parent Jacobian
  if (parent_link_ptr) {
    auto parent2body_pose{(joint2parent_pose_ * body2joint_pose_).inverse()};
    auto dtheta_body_dtheta_parent{Adjoint(parent2body_pose)};
    jacobian_ = dtheta_body_dtheta_parent * parent_link_ptr->jacobian_;
  } else {
    jacobian_.setZero();
  }

  // Add joint Jacobian
  auto joint2body_pose{body2joint_pose_.inverse()};
  auto dtheta_body_dtheta_joint{Adjoint(joint2body_pose)};
  int jacobian_idx = first_jacobian_index_;
  for (int direction = 0; direction < 6; ++direction) {
    if (free_directions_[direction]) {
      jacobian_.col(jacobian_idx) = dtheta_body_dtheta_joint.col(direction);
      jacobian_idx++;
    }
  }
  return true;
}

bool Link::CalculateGradientAndHessian() {
  if (!IsSetup(true)) return false;
  gradient_.setZero();
  hessian_.setZero();
  last_texture_reliability_weight_ = 1.0f;
  last_model_texture_reliability_weight_ = 1.0f;
  last_model_photometric_reliability_weight_ = 1.0f;
  last_model_contour_reliability_weight_ = 1.0f;
  last_model_ecc_reliability_weight_ = 1.0f;
  last_model_hypothesis_reliability_weight_ = 1.0f;
  last_video_ecc_reliability_weight_ = 1.0f;
  if (!use_reliability_weighted_fusion_) {
    for (auto &modality_ptr : modality_ptrs_) {
      gradient_ += modality_ptr->gradient();
      hessian_ += modality_ptr->hessian();
      if (auto texture_ptr =
              std::dynamic_pointer_cast<TextureModality>(modality_ptr)) {
        gradient_ += texture_ptr->model_texture_gradient();
        hessian_ += texture_ptr->model_texture_hessian();
        gradient_ += texture_ptr->model_photometric_gradient();
        hessian_ += texture_ptr->model_photometric_hessian();
        gradient_ += texture_ptr->model_contour_gradient();
        hessian_ += texture_ptr->model_contour_hessian();
        gradient_ += texture_ptr->model_ecc_gradient();
        hessian_ += texture_ptr->model_ecc_hessian();
        gradient_ += texture_ptr->model_hypothesis_gradient();
        hessian_ += texture_ptr->model_hypothesis_hessian();
        gradient_ += texture_ptr->video_ecc_gradient();
        hessian_ += texture_ptr->video_ecc_hessian();
      }
    }
    ApplyDirectionalTemporalPrior();
    return true;
  }

  // First assemble all non-texture modalities. They define the geometric
  // anchor against which appearance updates are checked.
  for (auto &modality_ptr : modality_ptrs_) {
    if (std::dynamic_pointer_cast<TextureModality>(modality_ptr)) continue;
    gradient_ += modality_ptr->gradient();
    hessian_ += modality_ptr->hessian();
  }
  Eigen::Matrix<float, 6, 1> tikhonov;
  tikhonov.head<3>().setConstant(1000.0f);
  tikhonov.tail<3>().setConstant(30000.0f);
  const auto inverse_sqrt_tikhonov{
      tikhonov.array().sqrt().inverse().matrix()};
  const Eigen::Matrix<float, 6, 6> anchor_information{-hessian_};
  const Eigen::Matrix<float, 6, 6> normalized_anchor_information{
      inverse_sqrt_tikhonov.asDiagonal() * anchor_information *
      inverse_sqrt_tikhonov.asDiagonal()};
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eigen_solver{
      normalized_anchor_information};
  const float anchor_minimum_information{
      eigen_solver.info() == Eigen::Success
          ? std::max(0.0f, eigen_solver.eigenvalues()(0))
          : 0.0f};
  Eigen::Matrix<float, 6, 6> regularized_anchor_information{
      anchor_information};
  regularized_anchor_information.diagonal() += tikhonov;
  const Eigen::Matrix<float, 6, 1> anchor_step{
      regularized_anchor_information.ldlt().solve(gradient_)};
  const auto normalized_anchor_step{
      tikhonov.array().sqrt().matrix().cwiseProduct(anchor_step)};

  for (auto &modality_ptr : modality_ptrs_) {
    auto texture_ptr{
        std::dynamic_pointer_cast<TextureModality>(modality_ptr)};
    if (!texture_ptr) continue;

    // Three 2D points provide the minimal six scalar pose constraints. The
    // 3-sigma residual gate follows directly from the final texture noise
    // model instead of introducing a dataset-specific score threshold.
    const float support_reliability{std::min(
        1.0f, float(texture_ptr->n_valid_gradient_points()) / 3.0f)};
    const float standard_deviation{
        texture_ptr->standard_deviations().empty()
            ? 1.0f
            : texture_ptr->standard_deviations().back()};
    const float residual_limit{3.0f * standard_deviation};
    const float residual_reliability{
        std::min(1.0f,
                 residual_limit /
                     std::max(residual_limit,
                              texture_ptr->mean_reprojection_error()))};

    const Eigen::Matrix<float, 6, 6> texture_information{
        -modality_ptr->hessian()};
    Eigen::Matrix<float, 6, 6> regularized_texture_information{
        texture_information};
    regularized_texture_information.diagonal() += tikhonov;
    const Eigen::Matrix<float, 6, 1> texture_step{
        regularized_texture_information.ldlt().solve(modality_ptr->gradient())};
    const auto normalized_texture_step{
        tikhonov.array().sqrt().matrix().cwiseProduct(texture_step)};
    float step_cosine{0.0f};
    const float step_norm_product{
        normalized_anchor_step.norm() * normalized_texture_step.norm()};
    if (step_norm_product > std::numeric_limits<float>::min())
      step_cosine = std::clamp(normalized_anchor_step.dot(
                                   normalized_texture_step) /
                                   step_norm_product,
                               -1.0f, 1.0f);
    const float anchor_reliability{anchor_minimum_information /
                                   (1.0f + anchor_minimum_information)};
    const float conflict_reliability{
        1.0f - anchor_reliability * std::max(0.0f, -step_cosine)};
    last_texture_reliability_weight_ =
        support_reliability * residual_reliability * conflict_reliability;
    gradient_ += last_texture_reliability_weight_ * modality_ptr->gradient();
    hessian_ += last_texture_reliability_weight_ * modality_ptr->hessian();

    // Absolute CAD terms are projected onto directions that remain weak after
    // geometric and video-keyframe fusion. This prevents synthetic/real
    // appearance differences from perturbing well-observed pose components.
    const auto add_complementary_term =
        [&](const Eigen::Matrix<float, 6, 1> &term_gradient,
            const Eigen::Matrix<float, 6, 6> &term_hessian,
            float support_reliability,
            float residual_reliability, bool apply_term) -> float {
      if (support_reliability <= 0.0f) return 0.0f;
      const Eigen::Matrix<float, 6, 6> fused_information{-hessian_};
      const Eigen::Matrix<float, 6, 6> normalized_fused_information{
          inverse_sqrt_tikhonov.asDiagonal() * fused_information *
          inverse_sqrt_tikhonov.asDiagonal()};
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> fused_solver{
          normalized_fused_information};
      if (fused_solver.info() != Eigen::Success) return 0.0f;
      Eigen::Matrix<float, 6, 6> complement_projector{
          Eigen::Matrix<float, 6, 6>::Zero()};
      for (int i = 0; i < 6; ++i) {
        const float eigenvalue{
            std::max(0.0f, fused_solver.eigenvalues()(i))};
        const float complement{1.0f / (1.0f + eigenvalue)};
        const auto direction{fused_solver.eigenvectors().col(i)};
        complement_projector +=
            complement * direction * direction.transpose();
      }

      const Eigen::Matrix<float, 6, 6> term_information{-term_hessian};
      const Eigen::Matrix<float, 6, 6> normalized_term_information{
          inverse_sqrt_tikhonov.asDiagonal() * term_information *
          inverse_sqrt_tikhonov.asDiagonal()};
      const Eigen::Matrix<float, 6, 1> normalized_term_gradient{
          inverse_sqrt_tikhonov.cwiseProduct(term_gradient)};
      const Eigen::Matrix<float, 6, 6> projected_term_information{
          complement_projector * normalized_term_information *
          complement_projector};
      const Eigen::Matrix<float, 6, 1> projected_term_gradient{
          complement_projector * normalized_term_gradient};

      Eigen::Matrix<float, 6, 6> regularized_term_information{
          term_information};
      regularized_term_information.diagonal() += tikhonov;
      const Eigen::Matrix<float, 6, 1> term_step{
          regularized_term_information.ldlt().solve(term_gradient)};
      const auto normalized_term_step{
          tikhonov.array().sqrt().matrix().cwiseProduct(term_step)};
      Eigen::Matrix<float, 6, 6> regularized_fused_information{
          fused_information};
      regularized_fused_information.diagonal() += tikhonov;
      const Eigen::Matrix<float, 6, 1> fused_step{
          regularized_fused_information.ldlt().solve(gradient_)};
      const auto normalized_fused_step{
          tikhonov.array().sqrt().matrix().cwiseProduct(fused_step)};
      float step_cosine{0.0f};
      const float step_norm_product{
          normalized_fused_step.norm() * normalized_term_step.norm()};
      if (step_norm_product > std::numeric_limits<float>::min())
        step_cosine = std::clamp(
            normalized_fused_step.dot(normalized_term_step) /
                step_norm_product,
            -1.0f, 1.0f);
      const float minimum_information{
          std::max(0.0f, fused_solver.eigenvalues()(0))};
      const float fused_reliability{minimum_information /
                                    (1.0f + minimum_information)};
      const float conflict_reliability{
          1.0f - fused_reliability * std::max(0.0f, -step_cosine)};
      const float reliability_weight{support_reliability *
                                     residual_reliability *
                                     conflict_reliability};
      const auto sqrt_tikhonov{tikhonov.array().sqrt().matrix()};
      if (apply_term) {
        gradient_ += reliability_weight *
                     sqrt_tikhonov.cwiseProduct(projected_term_gradient);
        hessian_ -= reliability_weight * sqrt_tikhonov.asDiagonal() *
                    projected_term_information * sqrt_tikhonov.asDiagonal();
      }
      return reliability_weight;
    };

    if (texture_ptr->use_model_texture_matching()) {
      struct ModelTextureCandidate {
        int n_valid_points = 0;
        float mean_reprojection_error = 0.0f;
        Eigen::Matrix<float, 6, 1> gradient =
            Eigen::Matrix<float, 6, 1>::Zero();
        Eigen::Matrix<float, 6, 6> hessian =
            Eigen::Matrix<float, 6, 6>::Zero();
        float reliability = 0.0f;
      };
      const auto evaluate_model_texture_source =
          [&](int n_valid_points, float mean_reprojection_error,
              const Eigen::Matrix<float, 6, 1> &source_gradient,
              const Eigen::Matrix<float, 6, 6> &source_hessian) {
            ModelTextureCandidate candidate;
            candidate.n_valid_points = n_valid_points;
            candidate.mean_reprojection_error = mean_reprojection_error;
            candidate.gradient = source_gradient;
            candidate.hessian = source_hessian;
            const float support_ratio{
                std::min(1.0f, float(n_valid_points) / 6.0f)};
            const float model_residual_ratio{
                std::min(1.0f,
                         residual_limit /
                             std::max(residual_limit,
                                      mean_reprojection_error))};
            candidate.reliability = add_complementary_term(
                source_gradient, source_hessian,
                support_ratio * support_ratio,
                model_residual_ratio * model_residual_ratio, false);
            return candidate;
          };
      const int n_orb_points{
          texture_ptr->n_valid_model_texture_orb_points()};
      const int n_disk_points{
          texture_ptr->n_valid_model_texture_disk_points()};
      if (n_orb_points == 0 || n_disk_points == 0) {
        const int n_points{n_orb_points > 0 ? n_orb_points : n_disk_points};
        const float mean_error{
            n_orb_points > 0
                ? texture_ptr->mean_model_texture_orb_reprojection_error()
                : texture_ptr->mean_model_texture_disk_reprojection_error()};
        const Eigen::Matrix<float, 6, 1> source_gradient{
            n_orb_points > 0 ? texture_ptr->model_texture_orb_gradient()
                             : texture_ptr->model_texture_disk_gradient()};
        const Eigen::Matrix<float, 6, 6> source_hessian{
            n_orb_points > 0 ? texture_ptr->model_texture_orb_hessian()
                             : texture_ptr->model_texture_disk_hessian()};
        const float support_ratio{
            std::min(1.0f, float(n_points) / 6.0f)};
        const float model_residual_ratio{
            std::min(1.0f,
                     residual_limit /
                         std::max(residual_limit, mean_error))};
        last_model_texture_reliability_weight_ = add_complementary_term(
            source_gradient, source_hessian, support_ratio * support_ratio,
            model_residual_ratio * model_residual_ratio, true);
      } else {
        const ModelTextureCandidate orb_candidate{
            evaluate_model_texture_source(
                n_orb_points,
                texture_ptr->mean_model_texture_orb_reprojection_error(),
                texture_ptr->model_texture_orb_gradient(),
                texture_ptr->model_texture_orb_hessian())};
        const ModelTextureCandidate disk_candidate{
            evaluate_model_texture_source(
                n_disk_points,
                texture_ptr->mean_model_texture_disk_reprojection_error(),
                texture_ptr->model_texture_disk_gradient(),
                texture_ptr->model_texture_disk_hessian())};
        const ModelTextureCandidate &selected_candidate{
            disk_candidate.reliability > orb_candidate.reliability
                ? disk_candidate
                : orb_candidate};
        const float support_ratio{std::min(
            1.0f, float(selected_candidate.n_valid_points) / 6.0f)};
        const float model_residual_ratio{
            std::min(1.0f,
                     residual_limit /
                         std::max(residual_limit,
                                  selected_candidate.mean_reprojection_error))};
        last_model_texture_reliability_weight_ = add_complementary_term(
            selected_candidate.gradient, selected_candidate.hessian,
            support_ratio * support_ratio,
            model_residual_ratio * model_residual_ratio, true);
      }
    }
    if (texture_ptr->use_model_photometric_tracking()) {
      const float support_reliability{std::min(
          1.0f,
          float(texture_ptr->n_valid_model_photometric_points()) / 30.0f)};
      const float photometric_residual_limit{30.0f};
      const float residual_ratio{
          std::min(1.0f,
                   photometric_residual_limit /
                       std::max(photometric_residual_limit,
                                texture_ptr
                                    ->mean_model_photometric_residual()))};
      last_model_photometric_reliability_weight_ = add_complementary_term(
          texture_ptr->model_photometric_gradient(),
          texture_ptr->model_photometric_hessian(), support_reliability,
          residual_ratio * residual_ratio, true);
    }
    if (texture_ptr->use_model_contour_tracking()) {
      const float support_ratio{std::min(
          1.0f, float(texture_ptr->n_valid_model_contour_points()) / 12.0f)};
      const float contour_residual_limit{6.0f};
      const float residual_ratio{
          std::min(1.0f,
                   contour_residual_limit /
                       std::max(contour_residual_limit,
                                texture_ptr->mean_model_contour_residual()))};
      last_model_contour_reliability_weight_ = add_complementary_term(
          texture_ptr->model_contour_gradient(),
          texture_ptr->model_contour_hessian(), support_ratio * support_ratio,
          residual_ratio * residual_ratio, true);
    }
    if (texture_ptr->use_model_ecc_tracking()) {
      const float support_ratio{std::min(
          1.0f, float(texture_ptr->n_valid_model_ecc_points()) / 12.0f)};
      const float residual_limit{6.0f};
      const float residual_ratio{
          std::min(1.0f,
                   residual_limit /
                       std::max(residual_limit,
                                texture_ptr->mean_model_ecc_residual()))};
      const float correlation_reliability{std::clamp(
          (texture_ptr->model_ecc_correlation() - 0.2f) / 0.5f, 0.0f,
          1.0f)};
      last_model_ecc_reliability_weight_ = add_complementary_term(
          texture_ptr->model_ecc_gradient(), texture_ptr->model_ecc_hessian(),
          support_ratio * support_ratio,
          residual_ratio * residual_ratio * correlation_reliability, true);
    }
    if (texture_ptr->use_model_hypothesis_search()) {
      last_model_hypothesis_reliability_weight_ = add_complementary_term(
          texture_ptr->model_hypothesis_gradient(),
          texture_ptr->model_hypothesis_hessian(),
          texture_ptr->model_hypothesis_confidence(), 1.0f, true);
    }
    if (texture_ptr->use_video_ecc_tracking()) {
      const float support_ratio{std::min(
          1.0f, float(texture_ptr->n_valid_video_ecc_points()) / 12.0f)};
      const float residual_limit{6.0f};
      const float residual_ratio{
          std::min(1.0f,
                   residual_limit /
                       std::max(residual_limit,
                                texture_ptr->mean_video_ecc_residual()))};
      const float correlation_reliability{std::clamp(
          (texture_ptr->video_ecc_correlation() - 0.4f) / 0.4f, 0.0f,
          1.0f)};
      last_video_ecc_reliability_weight_ = add_complementary_term(
          texture_ptr->video_ecc_gradient(), texture_ptr->video_ecc_hessian(),
          support_ratio * support_ratio,
          residual_ratio * residual_ratio * correlation_reliability, true);
    }
  }
  ApplyDirectionalTemporalPrior();
  return true;
}

void Link::ApplyDirectionalTemporalPrior() {
  if (!directional_temporal_prior_valid_ || !body_ptr_) return;

  // Marginalize translation before measuring rotational observability. The
  // fixed normalization matches the optimizer regularizer, so an eigenvalue
  // below one means that image evidence is weaker than the standard numerical
  // prior in that direction.
  const Eigen::Matrix<float, 6, 6> information{
      -0.5f * (hessian_ + hessian_.transpose())};
  Eigen::Matrix3f translation_information{
      information.bottomRightCorner<3, 3>()};
  translation_information.diagonal().array() += 30000.0f;
  const Eigen::Matrix3f rotational_information{
      information.topLeftCorner<3, 3>() -
      information.topRightCorner<3, 3>() *
          translation_information.ldlt().solve(
              information.bottomLeftCorner<3, 3>())};
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver{
      rotational_information / 1000.0f};
  if (eigen_solver.info() != Eigen::Success) return;

  Eigen::Matrix3f prior_information{Eigen::Matrix3f::Zero()};
  for (int i = 0; i < 3; ++i) {
    const float missing_information{
        std::max(0.0f, 1.0f - eigen_solver.eigenvalues()(i))};
    const Eigen::Vector3f direction{eigen_solver.eigenvectors().col(i)};
    prior_information +=
        (1000.0f * missing_information) * direction * direction.transpose();
  }
  if (prior_information.isZero()) return;

  const Transform3fA current2predicted_pose{
      body_ptr_->body2world_pose().inverse() * predicted_pose_};
  const Eigen::AngleAxisf rotation_error{
      current2predicted_pose.rotation().matrix()};
  Eigen::Matrix<float, 6, 1> target_variation{
      Eigen::Matrix<float, 6, 1>::Zero()};
  if (std::isfinite(rotation_error.angle()))
    target_variation.head<3>() = rotation_error.angle() * rotation_error.axis();

  gradient_.head<3>() += prior_information * target_variation.head<3>();
  hessian_.topLeftCorner<3, 3>() -= prior_information;
}

bool Link::AddToGradientAndHessian(
    const Eigen::Matrix<float, 6, 1> &gradient_summand,
    const Eigen::Matrix<float, 6, 6> &hessian_summand) {
  if (!IsSetup(true)) return false;

  gradient_ += gradient_summand;
  hessian_ += hessian_summand;
  return true;
}

bool Link::UpdatePoses(const std::shared_ptr<Link> &parent_link_ptr,
                       Eigen::VectorXf theta) {
  if (!IsSetup(true)) return false;

  // Extract variation vector for link
  Eigen::Matrix<float, 6, 1> theta_link;
  int jacobian_idx = first_jacobian_index_;
  for (int direction = 0; direction < 6; ++direction) {
    if (free_directions_[direction]) {
      theta_link(direction) = theta(jacobian_idx);
      jacobian_idx++;
    } else {
      theta_link(direction) = 0.0f;
    }
  }

  // Calculate pose variation
  Transform3fA pose_variation{Transform3fA::Identity()};
  pose_variation.translate(theta_link.tail<3>());
  pose_variation.rotate(Vector2Skewsymmetric(theta_link.head<3>()).exp());

  // Update poses
  if (parent_link_ptr) {
    if (fixed_body2joint_pose_)
      joint2parent_pose_ = joint2parent_pose_ * pose_variation;
    else
      body2joint_pose_ = pose_variation * body2joint_pose_;
    link2world_pose_ = parent_link_ptr->link2world_pose() * joint2parent_pose_ *
                       body2joint_pose_;
    if (body_ptr_) body_ptr_->set_body2world_pose(link2world_pose_);
  } else {
    link2world_pose_ = link2world_pose() * body2joint_pose_.inverse() *
                       pose_variation * body2joint_pose_;
    if (body_ptr_) body_ptr_->set_body2world_pose(link2world_pose_);
  }
  return true;
}

void Link::ResetJointPoses() {
  body2joint_pose_ = default_body2joint_pose_;
  joint2parent_pose_ = default_joint2parent_pose_;
}

int Link::DegreesOfFreedom() const {
  return int(
      std::count(begin(free_directions_), end(free_directions_), true));
}

const Eigen::Matrix<float, 6, 1> &Link::gradient() { return gradient_; }

const Eigen::Matrix<float, 6, 6> &Link::hessian() { return hessian_; }

const Eigen::Matrix<float, 6, Eigen::Dynamic> &Link::jacobian() {
  return jacobian_;
}

const std::string &Link::name() const { return name_; }

const std::filesystem::path &Link::metafile_path() const {
  return metafile_path_;
}

const std::shared_ptr<Body> &Link::body_ptr() const { return body_ptr_; }

const std::vector<std::shared_ptr<Modality>> &Link::modality_ptrs() const {
  return modality_ptrs_;
}

const std::vector<std::shared_ptr<Link>> &Link::child_link_ptrs() const {
  return child_link_ptrs_;
}

const Transform3fA &Link::body2joint_pose() const { return body2joint_pose_; }

const Transform3fA &Link::joint2parent_pose() const {
  return joint2parent_pose_;
}

const std::array<bool, 6> &Link::free_directions() const {
  return free_directions_;
}

bool Link::fixed_body2joint_pose() const { return fixed_body2joint_pose_; }

bool Link::use_reliability_weighted_fusion() const {
  return use_reliability_weighted_fusion_;
}

float Link::last_texture_reliability_weight() const {
  return last_texture_reliability_weight_;
}

float Link::last_model_texture_reliability_weight() const {
  return last_model_texture_reliability_weight_;
}

float Link::last_model_photometric_reliability_weight() const {
  return last_model_photometric_reliability_weight_;
}

float Link::last_model_contour_reliability_weight() const {
  return last_model_contour_reliability_weight_;
}

float Link::last_model_ecc_reliability_weight() const {
  return last_model_ecc_reliability_weight_;
}

float Link::last_model_hypothesis_reliability_weight() const {
  return last_model_hypothesis_reliability_weight_;
}

float Link::last_video_ecc_reliability_weight() const {
  return last_video_ecc_reliability_weight_;
}

const Transform3fA &Link::link2world_pose() const {
  if (body_ptr_)
    return body_ptr_->body2world_pose();
  else
    return link2world_pose_;
}

int Link::first_jacobian_index() const { return first_jacobian_index_; }

bool Link::set_up() const { return set_up_; }

bool Link::LoadMetaData() {
  // Open file storage from yaml
  cv::FileStorage fs;
  if (!OpenYamlFileStorage(metafile_path_, &fs)) return false;

  // Read parameters from yaml
  std::vector<char> free_directions_vector;
  ReadOptionalValueFromYaml(fs, "body2joint_pose", &body2joint_pose_);
  ReadOptionalValueFromYaml(fs, "joint2parent_pose", &joint2parent_pose_);
  ReadOptionalValueFromYaml(fs, "link2world_pose", &link2world_pose_);
  ReadOptionalValueFromYaml(fs, "free_directions", &free_directions_vector);
  ReadOptionalValueFromYaml(fs, "fixed_body2joint_pose",
                            &fixed_body2joint_pose_);
  default_body2joint_pose_ = body2joint_pose_;
  default_joint2parent_pose_ = joint2parent_pose_;

  // Process parameters
  if (free_directions_vector.size() == 6) {
    for (int i = 0; i < 6; ++i)
      free_directions_[i] = bool(free_directions_vector[i]);
  } else {
    std::cerr << "Parameter \"free_directions\" in " << metafile_path_
              << " does not contain 6 boolean values." << std::endl;
    return false;
  }

  fs.release();
  return true;
}

bool Link::IsSetup(bool check_jacobian_defined) {
  if (!set_up_) {
    std::cerr << "Set up link " << name_ << " first" << std::endl;
    return false;
  }
  if (check_jacobian_defined && !jacobian_defined_) {
    std::cerr << "Define jacobian of link " << name_ << " first" << std::endl;
    return false;
  }
  return true;
}

Eigen::Matrix<float, 6, 6> Link::Adjoint(const Transform3fA &pose) {
  Eigen::Matrix<float, 6, 6> m{Eigen::Matrix<float, 6, 6>::Zero()};
  m.topLeftCorner<3, 3>() = pose.rotation();
  m.bottomLeftCorner<3, 3>() =
      Vector2Skewsymmetric(pose.translation()) * pose.rotation();
  m.bottomRightCorner<3, 3>() = pose.rotation();
  return m;
}

}  // namespace m3t
