! Copyright (c), The Regents of the University of California
! Terms of use are as specified in LICENSE.txt
module trainable_engine_m
  !! Define an abstraction that supports training a neural network

  use sourcery_m, only : string_t
  use inference_engine_m_, only : inference_engine_t
  use outputs_m, only : outputs_t
  use differentiable_activation_strategy_m, only : differentiable_activation_strategy_t
  use kind_parameters_m, only : rkind
  use inputs_m, only :  inputs_t
  use expected_outputs_m, only : expected_outputs_t 
  use mini_batch_m, only : mini_batch_t
  implicit none

  private
  public :: trainable_engine_t

  type trainable_engine_t
    !! Encapsulate the information needed to perform training
    private
    type(string_t), allocatable :: metadata_(:)
    real(rkind), allocatable :: w(:,:,:) ! weights
    real(rkind), allocatable :: b(:,:) ! biases
    integer, allocatable :: n(:) ! nuerons per layer
    class(differentiable_activation_strategy_t), allocatable :: differentiable_activation_strategy_ 
  contains
    procedure :: assert_consistent
    procedure :: train
    procedure :: infer
    procedure :: num_layers
    procedure :: num_inputs
    procedure :: to_inference_engine
  end type

  integer, parameter :: input_layer = 0

  interface trainable_engine_t

    pure module function construct_from_padded_arrays(nodes, weights, biases, differentiable_activation_strategy, metadata) &
    result(trainable_engine)
      implicit none
      integer, intent(in) :: nodes(input_layer:)
      real(rkind), intent(in)  :: weights(:,:,:), biases(:,:)
      class(differentiable_activation_strategy_t), intent(in) :: differentiable_activation_strategy
      type(string_t), intent(in) :: metadata(:)
      type(trainable_engine_t) trainable_engine
    end function

  end interface

  interface

    pure module subroutine assert_consistent(self)
      implicit none
      class(trainable_engine_t), intent(in) :: self
    end subroutine

    pure module subroutine train(self, mini_batches)
      implicit none
      class(trainable_engine_t), intent(inout) :: self
      type(mini_batch_t), intent(in) :: mini_batches(:)
    end subroutine

    elemental module function infer(self, inputs) result(outputs)
      implicit none
      class(trainable_engine_t), intent(in) :: self
      type(inputs_t), intent(in) :: inputs
      type(outputs_t) outputs
    end function
    
    elemental module function num_inputs(self) result(n_in)
      implicit none
      class(trainable_engine_t), intent(in) :: self
      integer n_in
    end function

    elemental module function num_layers(self) result(n_layers)
      implicit none
      class(trainable_engine_t), intent(in) :: self
      integer n_layers
    end function

    pure module function to_inference_engine(self) result(inference_engine)
      implicit none
      class(trainable_engine_t), intent(in) :: self
      type(inference_engine_t) :: inference_engine
    end function

  end interface

end module trainable_engine_m
