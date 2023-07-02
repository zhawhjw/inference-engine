! Copyright (c), The Regents of the University of California
! Terms of use are as specified in LICENSE.txt
module skip_connections_test_m
  !! Define inference tests and procedures required for reporting results

  ! External dependencies
  use assert_m, only : assert
  use kind_parameters_m, only : rkind
  use string_m, only : string_t
  use test_m, only : test_t
  use test_result_m, only : test_result_t

  ! Internal dependencies
  use inference_engine_m, only : inference_engine_t, inference_strategy_t, matmul_t, concurrent_dot_products_t, outputs_t

  implicit none

  private
  public :: skip_connections_test_t

  type, extends(test_t) :: skip_connections_test_t
  contains
    procedure, nopass :: subject
    procedure, nopass :: results
  end type

contains

  pure function subject() result(specimen)
    character(len=:), allocatable :: specimen
    specimen = "An inference_engine_t object encoding 2nd-input-AND-NOT-1st-input logic with skip connections"
  end function

  function results() result(test_results)
    type(test_result_t), allocatable :: test_results(:)

      character(len=*), parameter :: longest_description = &
            "mapping (false,false) to false using the concurrent_dot_products_t() inference strategy"

      associate( &
        descriptions => &
          [ character(len=len(longest_description)) :: &
            "mapping (true,true) to false using the concurrent_dot_products_t() inference strategy", &
            "mapping (true,false) to false using the concurrent_dot_products_t() inference strategy", &
            "mapping (false,true) to true using the concurrent_dot_products_t() inference strategy", &
            "mapping (false,false) to false using the concurrent_dot_products_t() inference strategy", &
            "mapping (true,true) to false using the matmul_t() inference strategy", &
            "mapping (true,false) to false using the matmul_t() inference strategy", &
            "mapping (false,true) to true using the matmul_t() inference strategy", &
            "mapping (false,false) to false using the matmul_t() inference strategy" &
          ], &
        outcomes => [not_1st_and_2nd_truth_table(concurrent_dot_products_t()), not_1st_and_2nd_truth_table(matmul_t()) ] &
       )
       call assert(size(descriptions) == size(outcomes), "skip_connections_test_m(results): size(descriptions) == size(outcomes)")
       test_results = test_result_t(descriptions, outcomes)
     end associate
    
  end function


  function not_1st_and_2nd_truth_table(inference_strategy) result(test_passes)
    class(inference_strategy_t), intent(in) :: inference_strategy
    logical, allocatable :: test_passes(:)
    type(inference_engine_t) inference_engine
    real(rkind), parameter :: tolerance = 1.E-08_rkind, false = 0._rkind, true = 1._rkind
    integer, parameter :: n_in = 2 ! number of inputs
    integer, parameter :: n_out = 1 ! number of outputs
    integer, parameter :: neurons = 2 ! number of neurons per layer
    integer, parameter :: n_hidden = 2 ! number of hidden layers 
    integer i 
      
    inference_engine = inference_engine_t( &
      metadata = [string_t("XOR"), string_t("Damian Rouson"), string_t("2023-02-18"), string_t("step"), string_t("true")], &
      input_weights  = real(reshape([1,0,0,1], [n_in, neurons]), rkind), &
      hidden_weights = real(reshape([(0., i=1,neurons*neurons)], [neurons, neurons, n_hidden-1]), rkind), &
      output_weights = real(reshape([-2,1], [n_out, neurons]), rkind), &
      biases = reshape([real(rkind):: 0.,0, 0.,0.], [neurons, n_hidden]), &
      output_biases = [real(rkind):: 0.] &
    )
    call assert(inference_engine%skip(), "skip_connections_test_m(not_1st_and_2nd_truth_table): skip")

    block
      type(outputs_t)  true_true, true_false, false_true, false_false

      true_true = inference_engine%infer([true,true], inference_strategy)
      true_false = inference_engine%infer([true,false], inference_strategy)
      false_true = inference_engine%infer([false,true], inference_strategy)
      false_false = inference_engine%infer([false,false], inference_strategy) 

      associate( &
        true_true_outputs => true_true%outputs(), &
        true_false_outputs => true_false%outputs(), &
        false_true_outputs => false_true%outputs(), &
        false_false_outputs => false_false%outputs() &
      )
        test_passes = [ &
          size(true_true_outputs)==1 .and. abs(true_true_outputs(1) - false) < tolerance, &
          size(true_false_outputs)==1 .and. abs(true_false_outputs(1) - false) < tolerance,  &
          size(false_true_outputs)==1 .and. abs(false_true_outputs(1) - true) < tolerance, &
          size(false_false_outputs)==1 .and. abs(false_false_outputs(1) - false) < tolerance  &
        ]
      end associate
    end block

  end function

end module skip_connections_test_m