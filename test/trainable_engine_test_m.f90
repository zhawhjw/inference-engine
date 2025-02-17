! Copyright (c), The Regents of the University of California
! Terms of use are as specified in LICENSE.txt
module trainable_engine_test_m
  !! Define inference tests and procedures required for reporting results

  ! External dependencies
  use assert_m, only : assert
  use intrinsic_array_m, only : intrinsic_array_t
  use kind_parameters_m, only : rkind
  use string_m, only : string_t
  use test_m, only : test_t
  use test_result_m, only : test_result_t

  ! Internal dependencies
  use inference_engine_m, only : &
    trainable_engine_t, inputs_t, outputs_t, expected_outputs_t, sigmoid_t, input_output_pair_t, mini_batch_t

  implicit none

  private
  public :: trainable_engine_test_t

  type, extends(test_t) :: trainable_engine_test_t
  contains
    procedure, nopass :: subject
    procedure, nopass :: results
  end type

  real(rkind), parameter :: false = 0._rkind, true = 1._rkind

  abstract interface

    function map_i(inputs) result(expected_outputs)
      import inputs_t, expected_outputs_t
      type(inputs_t), intent(in) :: inputs
      type(expected_outputs_t) expected_outputs
    end function

  end interface

contains

  pure function subject() result(specimen)
    character(len=:), allocatable :: specimen
    specimen = "A trainable_engine_t" 
  end function

  function results() result(test_results)
    type(test_result_t), allocatable :: test_results(:)

    character(len=*), parameter :: longest_description = &
        "learning the mapping (false,false) -> false with 2 hidden layers trained on symmetric OR-gate data and random weights"

    associate( &
      descriptions => &
      [ character(len=len(longest_description)) :: &
        "learning the mapping (true,true) -> true with 2 hidden layers trained on skewed AND-gate data"                         ,&
        "learning the mapping (false,true) -> false with 2 hidden layers trained on skewed AND-gate data"                       ,&
        "learning the mapping (true,false) -> false with 2 hidden layers trained on skewed AND-gate data"                       ,&
        "learning the mapping (false,false) -> false with 2 hidden layers trained on skewed AND-gate data"                      ,&
        "learning the mapping (true,true) -> false with 2 hidden layers trained on skewed NOT-AND-gate data"                    ,&
        "learning the mapping (false,true) -> true with 2 hidden layers trained on skewed NOT-AND-gate data"                    ,&
        "learning the mapping (true,false) -> true with 2 hidden layers trained on skewed NOT-AND-gate data"                    ,&
        "learning the mapping (false,false) -> true with 2 hidden layers trained on skewed NOT-AND-gate data"                   ,&
        "learning the mapping (true,true) -> true with 2 hidden layers trained on symmetric OR-gate data and random weights"    ,&
        "learning the mapping (false,true) -> true with 2 hidden layers trained on symmetric OR-gate data and random weights"   ,&
        "learning the mapping (true,false) -> true with 2 hidden layers trained on symmetric OR-gate data and random weights"   ,&
        "learning the mapping (false,false) -> false with 2 hidden layers trained on symmetric OR-gate data and random weights" ,&
        "learning the mapping (true,true) -> false with 2 hidden layers trained on symmetric XOR-gate data and random weights"  ,&
        "learning the mapping (false,true) -> true with 2 hidden layers trained on symmetric XOR-gate data and random weights"  ,&
        "learning the mapping (true,false) -> true with 2 hidden layers trained on symmetric XOR-gate data and random weights"  ,&
        "learning the mapping (false,false) -> false with 2 hidden layers trained on symmetric XOR-gate data and random weights" &
      ], outcomes => [ &
        and_gate_with_skewed_training_data(), &
        not_and_gate_with_skewed_training_data(), &
        or_gate_with_random_weights(), &
        xor_gate_with_random_weights() &
      ] &
    )
      associate(d => size(descriptions), o => size(outcomes))
        call assert(d == o, "trainable_engine_test_m(results): size(descriptions) == size(outcomes)", intrinsic_array_t([d,o]))
      end associate
      test_results = test_result_t(descriptions, outcomes)
    end associate
  end function

  subroutine print_truth_table(gate_name, gate_function_ptr, test_inputs, actual_outputs)
    !! Usage: 
    !!   procedure(map_i), pointer :: xor_ptr
    !!   xor_ptr => xor
    !!   call print_truth_table("XOR", xor_ptr, test_inputs, actual_outputs)
    character(len=*), intent(in) :: gate_name
    procedure(map_i), intent(in), pointer :: gate_function_ptr
    type(inputs_t), intent(in) :: test_inputs(:)
    type(outputs_t), intent(in) :: actual_outputs(:)
    type(expected_outputs_t) expected_outputs
    integer i

    call assert( size(test_inputs) == size(actual_outputs), &
      "trainable_engine_test_m(print_truth_table): size(test_inputs) == size(actual_outputs)")

    print *,"_______" // gate_name // "_______"

    do i = 1, size(test_inputs)
      expected_outputs = gate_function_ptr(test_inputs(i))
      print *,test_inputs(i)%values(), "-->", expected_outputs%outputs(), ":", actual_outputs(i)%outputs()
    end do
  end subroutine

  function two_zeroed_hidden_layers() result(trainable_engine)
    type(trainable_engine_t) trainable_engine
    integer, parameter :: inputs = 2, outputs = 1, hidden = 3 ! number of neurons in input, output, and hidden layers
    integer, parameter :: neurons(*) = [inputs, hidden, hidden, outputs] ! neurons per layer
    integer, parameter :: max_neurons = maxval(neurons), layers=size(neurons) ! max layer width, number of layers
    real(rkind) w(max_neurons, max_neurons, layers-1), b(max_neurons, max_neurons)

    w = 0.
    b = 0.

    trainable_engine = trainable_engine_t( &
      nodes = neurons, weights = w, biases = b, differentiable_activation_strategy = sigmoid_t(), metadata = &
      [string_t("2-hide|3-wide"), string_t("Damian Rouson"), string_t("2023-06-30"), string_t("sigmoid"), string_t("false")] &
    )   
  end function

  function two_random_hidden_layers() result(trainable_engine)
    type(trainable_engine_t) trainable_engine
    integer, parameter :: inputs = 2, outputs = 1, hidden = 3 ! number of neurons in input, output, and hidden layers
    integer, parameter :: neurons(*) = [inputs, hidden, hidden, outputs] ! neurons per layer
    integer, parameter :: max_neurons = maxval(neurons), layers=size(neurons) ! max layer width, number of layers
    real(rkind) w(max_neurons, max_neurons, layers-1), b(max_neurons, max_neurons)

    call random_number(b)
    call random_number(w)

    trainable_engine = trainable_engine_t( &
      nodes = neurons, weights = w, biases = b, differentiable_activation_strategy = sigmoid_t(), metadata = &
      [string_t("2-hide|3-wide"), string_t("Damian Rouson"), string_t("2023-06-30"), string_t("sigmoid"), string_t("false")] &
    )   
  end function

  function and_gate_with_skewed_training_data() result(test_passes)
    logical, allocatable :: test_passes(:)
    type(mini_batch_t), allocatable :: mini_batches(:)
    type(inputs_t), allocatable :: training_inputs(:,:), tmp(:), test_inputs(:)
    type(expected_outputs_t), allocatable :: training_outputs(:,:), expected_test_outputs(:), tmp2(:)
    type(trainable_engine_t) trainable_engine
    type(outputs_t), allocatable :: actual_outputs(:)
    real(rkind), parameter :: tolerance = 1.E-02_rkind
    real(rkind), allocatable :: harvest(:,:,:)
    integer, parameter :: num_inputs=2, mini_batch_size = 1, num_iterations=20000
    integer batch, iter, i

    allocate(harvest(num_inputs, mini_batch_size, num_iterations))
    call random_number(harvest)
    harvest = 2.*(harvest - 0.5) ! skew toward more input values being true

    ! The following temporary copies are required by gfortran bug 100650 and possibly 49324
    ! See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100650 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=49324
    tmp = [([(inputs_t(merge(true, false, harvest(:,batch,iter) < 0.5E0)), batch=1, mini_batch_size)], iter=1, num_iterations)]
    training_inputs = reshape(tmp, [mini_batch_size, num_iterations])

    tmp2 = [([(and(training_inputs(batch, iter)), batch = 1, mini_batch_size)], iter = 1, num_iterations )]
    training_outputs = reshape(tmp2, [mini_batch_size, num_iterations])

    mini_batches = [(mini_batch_t(input_output_pair_t(training_inputs(:,iter), training_outputs(:,iter))), iter=1, num_iterations)]        
    trainable_engine = two_zeroed_hidden_layers()

    call trainable_engine%train(mini_batches)

    test_inputs = [inputs_t([true,true]), inputs_t([false,true]), inputs_t([true,false]), inputs_t([false,false])]
    expected_test_outputs = [(and(test_inputs(i)), i=1, size(test_inputs))]
    actual_outputs = trainable_engine%infer(test_inputs)
    test_passes = [(abs(actual_outputs(i)%outputs() - expected_test_outputs(i)%outputs()) < tolerance, i=1, size(actual_outputs))]

  contains

    elemental function and(inputs_object) result(expected_outputs_object)
      type(inputs_t), intent(in) :: inputs_object 
      type(expected_outputs_t) expected_outputs_object 
      expected_outputs_object = expected_outputs_t([merge(true, false, sum(inputs_object%values()) > 1.99_rkind)])
    end function

  end function

  function not_and_gate_with_skewed_training_data() result(test_passes)
    logical, allocatable :: test_passes(:)
    type(mini_batch_t), allocatable :: mini_batches(:)
    type(inputs_t), allocatable :: training_inputs(:,:), tmp(:), test_inputs(:)
    type(expected_outputs_t), allocatable :: training_outputs(:,:), expected_test_outputs(:), tmp2(:)
    type(trainable_engine_t) trainable_engine
    type(outputs_t), allocatable :: actual_outputs(:)
    real(rkind), parameter :: tolerance = 1.E-02_rkind
    real(rkind), allocatable :: harvest(:,:,:)
    integer, parameter :: num_inputs=2, mini_batch_size = 1, num_iterations=30000
    integer batch, iter, i

    allocate(harvest(num_inputs, mini_batch_size, num_iterations))
    call random_number(harvest)
    harvest = 2.*(harvest - 0.5) ! skew toward more input values being true

    ! The following temporary copies are required by gfortran bug 100650 and possibly 49324
    ! See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100650 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=49324
    tmp = [([(inputs_t(merge(true, false, harvest(:,batch,iter) < 0.5E0)), batch=1, mini_batch_size)], iter=1, num_iterations)]
    training_inputs = reshape(tmp, [mini_batch_size, num_iterations])

    tmp2 = [([(not_and(training_inputs(batch, iter)), batch = 1, mini_batch_size)], iter = 1, num_iterations )]
    training_outputs = reshape(tmp2, [mini_batch_size, num_iterations])

    mini_batches = [(mini_batch_t(input_output_pair_t(training_inputs(:,iter), training_outputs(:,iter))), iter=1, num_iterations)]        
    trainable_engine = two_zeroed_hidden_layers()

    call trainable_engine%train(mini_batches)

    test_inputs = [inputs_t([true,true]), inputs_t([false,true]), inputs_t([true,false]), inputs_t([false,false])]
    expected_test_outputs = [(not_and(test_inputs(i)), i=1, size(test_inputs))]
    actual_outputs = trainable_engine%infer(test_inputs)
    test_passes = [(abs(actual_outputs(i)%outputs() - expected_test_outputs(i)%outputs()) < tolerance, i=1, size(actual_outputs))]

  contains
    
    function not_and(inputs) result(expected_outputs)
       type(inputs_t), intent(in) :: inputs
       type(expected_outputs_t) expected_outputs
       expected_outputs = expected_outputs_t([merge(true, false, sum(inputs%values()) < 2.)])
    end function

  end function

  function or_gate_with_random_weights() result(test_passes)
    logical, allocatable :: test_passes(:)
    type(mini_batch_t), allocatable :: mini_batches(:)
    type(inputs_t), allocatable :: training_inputs(:,:), tmp(:), test_inputs(:)
    type(expected_outputs_t), allocatable :: training_outputs(:,:), expected_test_outputs(:), tmp2(:)
    type(trainable_engine_t) trainable_engine
    type(outputs_t), allocatable :: actual_outputs(:)
    real(rkind), parameter :: tolerance = 1.E-02_rkind
    real(rkind), allocatable :: harvest(:,:,:)
    integer, parameter :: num_inputs=2, mini_batch_size = 1, num_iterations=50000
    integer batch, iter, i

    allocate(harvest(num_inputs, mini_batch_size, num_iterations))
    call random_number(harvest)

    ! The following temporary copies are required by gfortran bug 100650 and possibly 49324
    ! See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100650 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=49324
    tmp = [([(inputs_t(merge(true, false, harvest(:,batch,iter) < 0.5E0)), batch=1, mini_batch_size)], iter=1, num_iterations)]
    training_inputs = reshape(tmp, [mini_batch_size, num_iterations])

    tmp2 = [([(or(training_inputs(batch, iter)), batch = 1, mini_batch_size)], iter = 1, num_iterations )]
    training_outputs = reshape(tmp2, [mini_batch_size, num_iterations])

    mini_batches = [(mini_batch_t(input_output_pair_t(training_inputs(:,iter), training_outputs(:,iter))), iter=1, num_iterations)]        
    trainable_engine = two_random_hidden_layers()

    call trainable_engine%train(mini_batches)

    test_inputs = [inputs_t([true,true]), inputs_t([false,true]), inputs_t([true,false]), inputs_t([false,false])]
    expected_test_outputs = [(or(test_inputs(i)), i=1, size(test_inputs))]
    actual_outputs = trainable_engine%infer(test_inputs)
    test_passes = [(abs(actual_outputs(i)%outputs() - expected_test_outputs(i)%outputs()) < tolerance, i=1, size(actual_outputs))]

  contains
    
    function or(inputs) result(expected_outputs)
       type(inputs_t), intent(in) :: inputs
       type(expected_outputs_t) expected_outputs
       expected_outputs = expected_outputs_t([merge(true, false, sum(inputs%values()) > 0.99)])
    end function

  end function

  function xor_gate_with_random_weights() result(test_passes)
    logical, allocatable :: test_passes(:)
    type(mini_batch_t), allocatable :: mini_batches(:)
    type(inputs_t), allocatable :: training_inputs(:,:), tmp(:), test_inputs(:)
    type(expected_outputs_t), allocatable :: training_outputs(:,:), expected_test_outputs(:), tmp2(:)
    type(trainable_engine_t) trainable_engine
    type(outputs_t), allocatable :: actual_outputs(:)
    real(rkind), parameter :: tolerance = 1.E-02_rkind
    real(rkind), allocatable :: harvest(:,:,:)
    integer, parameter :: num_inputs=2, mini_batch_size = 1, num_iterations=400000
    integer batch, iter, i

    allocate(harvest(num_inputs, mini_batch_size, num_iterations))
    call random_number(harvest)

    ! The following temporary copies are required by gfortran bug 100650 and possibly 49324
    ! See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100650 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=49324
    tmp = [([(inputs_t(merge(true, false, harvest(:,batch,iter) < 0.5E0)), batch=1, mini_batch_size)], iter=1, num_iterations)]
    training_inputs = reshape(tmp, [mini_batch_size, num_iterations])

    tmp2 = [([(xor(training_inputs(batch, iter)), batch = 1, mini_batch_size)], iter = 1, num_iterations )]
    training_outputs = reshape(tmp2, [mini_batch_size, num_iterations])

    mini_batches = [(mini_batch_t(input_output_pair_t(training_inputs(:,iter), training_outputs(:,iter))), iter=1, num_iterations)]        
    trainable_engine = two_random_hidden_layers()

    call trainable_engine%train(mini_batches)

    test_inputs = [inputs_t([true,true]), inputs_t([false,true]), inputs_t([true,false]), inputs_t([false,false])]
    expected_test_outputs = [(xor(test_inputs(i)), i=1, size(test_inputs))]
    actual_outputs = trainable_engine%infer(test_inputs)
    test_passes = [(abs(actual_outputs(i)%outputs() - expected_test_outputs(i)%outputs()) < tolerance, i=1, size(actual_outputs))]

  contains
    
    function xor(inputs) result(expected_outputs)
      type(inputs_t), intent(in) :: inputs
      type(expected_outputs_t) expected_outputs
      associate(sum_inputs => sum(inputs%values()))
       expected_outputs = expected_outputs_t([merge(true, false, sum_inputs > 0.99 .and. sum_inputs < 1.01)])
      end associate
    end function

  end function

end module trainable_engine_test_m