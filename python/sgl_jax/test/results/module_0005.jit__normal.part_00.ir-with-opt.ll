; ModuleID = '__compute_module_part_00'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@broadcast_multiply_fusion.clone_parallel_bounds = private unnamed_addr constant [18 x [2 x [2 x i64]]] [[2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 8330, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 8330, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 8330, i64 10000]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @broadcast_multiply_fusion.clone(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %tid_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x = load i64, ptr %tids, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %arg2_gep = getelementptr i8, ptr %args, i64 32
  %arg2 = load ptr, ptr %arg2_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %lo_dim_0_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_multiply_fusion.clone_parallel_bounds, i64 0, i64 %tid_x, i64 0, i64 0
  %up_dim_0_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_multiply_fusion.clone_parallel_bounds, i64 0, i64 %tid_x, i64 0, i64 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 16
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 8
  %lo_dim_1_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_multiply_fusion.clone_parallel_bounds, i64 0, i64 %tid_x, i64 1, i64 0
  %up_dim_1_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_multiply_fusion.clone_parallel_bounds, i64 0, i64 %tid_x, i64 1, i64 1
  %lo_dim_1 = load i64, ptr %lo_dim_1_gep, align 16
  %up_dim_1 = load i64, ptr %up_dim_1_gep, align 8
  %.not5 = icmp ult i64 %lo_dim_0, %up_dim_0
  %.not13 = icmp ult i64 %lo_dim_1, %up_dim_1
  %or.cond = select i1 %.not5, i1 %.not13, i1 false
  br i1 %or.cond, label %broadcast_multiply_fusion.clone.loop_header.dim.1.preheader.us, label %return

broadcast_multiply_fusion.clone.loop_header.dim.1.preheader.us: ; preds = %1, %broadcast_multiply_fusion.clone.loop_header.dim.1.broadcast_multiply_fusion.clone.loop_exit.dim.1_crit_edge.us
  %broadcast_multiply_fusion.clone.invar_address.dim.0.06.us = phi i64 [ %invar.inc.us, %broadcast_multiply_fusion.clone.loop_header.dim.1.broadcast_multiply_fusion.clone.loop_exit.dim.1_crit_edge.us ], [ %lo_dim_0, %1 ]
  br label %vector.ph

vector.ph:                                        ; preds = %middle.block, %broadcast_multiply_fusion.clone.loop_header.dim.1.preheader.us
  %broadcast_multiply_fusion.clone.invar_address.dim.1.04.us = phi i64 [ %lo_dim_1, %broadcast_multiply_fusion.clone.loop_header.dim.1.preheader.us ], [ %invar.inc5.us, %middle.block ]
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %2 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg0, i64 0, i64 %broadcast_multiply_fusion.clone.invar_address.dim.0.06.us, i64 %broadcast_multiply_fusion.clone.invar_address.dim.1.04.us, i64 %index
  %wide.load = load <8 x i32>, ptr %2, align 32, !invariant.load !1, !noalias !4
  %3 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_multiply_fusion.clone.invar_address.dim.0.06.us, i64 %broadcast_multiply_fusion.clone.invar_address.dim.1.04.us, i64 %index
  %wide.load10 = load <8 x i32>, ptr %3, align 32, !invariant.load !1, !noalias !4
  %4 = xor <8 x i32> %wide.load10, %wide.load
  %5 = lshr <8 x i32> %4, splat (i32 9)
  %6 = or disjoint <8 x i32> %5, splat (i32 1065353216)
  %7 = bitcast <8 x i32> %6 to <8 x float>
  %8 = fadd <8 x float> %7, splat (float -1.000000e+00)
  %9 = fmul <8 x float> %8, splat (float 2.000000e+00)
  %10 = fadd <8 x float> %9, splat (float 0xBFEFFFFFE0000000)
  %11 = tail call <8 x float> @llvm.maximum.v8f32(<8 x float> %10, <8 x float> splat (float 0xBFEFFFFFE0000000))
  %12 = tail call <8 x float> @llvm.fabs.v8f32(<8 x float> %11)
  %13 = fcmp oeq <8 x float> %12, splat (float 1.000000e+00)
  %14 = fneg <8 x float> %11
  %15 = fmul <8 x float> %11, %14
  %16 = fadd <8 x float> %15, splat (float 1.000000e+00)
  %log_f32.i = fcmp ule <8 x float> %16, zeroinitializer
  %log_f321.i = sext <8 x i1> %log_f32.i to <8 x i32>
  %log_f322.i = bitcast <8 x i32> %log_f321.i to <8 x float>
  %log_f323.i = fcmp oeq <8 x float> %16, zeroinitializer
  %log_f324.i = sext <8 x i1> %log_f323.i to <8 x i32>
  %log_f325.i = bitcast <8 x i32> %log_f324.i to <8 x float>
  %log_f326.i = fcmp oeq <8 x float> %16, splat (float 0x7FF0000000000000)
  %log_f327.i = sext <8 x i1> %log_f326.i to <8 x i32>
  %log_f328.i = bitcast <8 x i32> %log_f327.i to <8 x float>
  %17 = fcmp uge <8 x float> splat (float 0x3810000000000000), %16
  %18 = select <8 x i1> %17, <8 x float> splat (float 0x3810000000000000), <8 x float> %16
  %19 = bitcast <8 x float> %18 to <8 x i32>
  %20 = lshr <8 x i32> %19, splat (i32 23)
  %log_f329.i = bitcast <8 x float> %18 to <8 x i32>
  %log_f3210.i = and <8 x i32> %log_f329.i, splat (i32 -2139095041)
  %21 = bitcast <8 x i32> %log_f3210.i to <8 x float>
  %log_f3212.i = or <8 x i32> %log_f3210.i, splat (i32 1056964608)
  %log_f3213.i = bitcast <8 x i32> %log_f3212.i to <8 x float>
  %22 = sub <8 x i32> %20, splat (i32 127)
  %23 = sitofp <8 x i32> %22 to <8 x float>
  %log_f3214.i = fadd <8 x float> splat (float 1.000000e+00), %23
  %log_f3215.i = fcmp olt <8 x float> %log_f3213.i, splat (float 0x3FE6A09E60000000)
  %log_f3216.i = sext <8 x i1> %log_f3215.i to <8 x i32>
  %log_f3217.i = bitcast <8 x i32> %log_f3216.i to <8 x float>
  %log_f3220.i = and <8 x i32> %log_f3212.i, %log_f3216.i
  %24 = bitcast <8 x i32> %log_f3220.i to <8 x float>
  %25 = fsub <8 x float> %log_f3213.i, splat (float 1.000000e+00)
  %log_f3222.i = and <8 x i32> %log_f3216.i, splat (i32 1065353216)
  %26 = bitcast <8 x i32> %log_f3222.i to <8 x float>
  %27 = fsub <8 x float> %log_f3214.i, %26
  %log_f3223.i = fadd <8 x float> %25, %24
  %log_f3224.i = fmul <8 x float> %log_f3223.i, %log_f3223.i
  %log_f3225.i = fmul <8 x float> %log_f3224.i, %log_f3223.i
  %log_f3226.i = fmul <8 x float> %log_f3223.i, splat (float 0x3FB2043760000000)
  %log_f3227.i = fadd <8 x float> splat (float 0xBFBD7A3700000000), %log_f3226.i
  %log_f3228.i = fmul <8 x float> %log_f3223.i, splat (float 0xBFBFCBA9E0000000)
  %log_f3229.i = fadd <8 x float> splat (float 0x3FC23D37E0000000), %log_f3228.i
  %log_f3230.i = fmul <8 x float> %log_f3223.i, splat (float 0x3FC999D580000000)
  %log_f3231.i = fadd <8 x float> splat (float 0xBFCFFFFF80000000), %log_f3230.i
  %log_f3232.i = fmul <8 x float> %log_f3227.i, %log_f3223.i
  %log_f3233.i = fadd <8 x float> splat (float 0x3FBDE4A340000000), %log_f3232.i
  %log_f3234.i = fmul <8 x float> %log_f3229.i, %log_f3223.i
  %log_f3235.i = fadd <8 x float> splat (float 0xBFC555CA00000000), %log_f3234.i
  %log_f3236.i = fmul <8 x float> %log_f3231.i, %log_f3223.i
  %log_f3237.i = fadd <8 x float> splat (float 0x3FD5555540000000), %log_f3236.i
  %log_f3238.i = fmul <8 x float> %log_f3233.i, %log_f3225.i
  %log_f3239.i = fadd <8 x float> %log_f3235.i, %log_f3238.i
  %log_f3240.i = fmul <8 x float> %log_f3239.i, %log_f3225.i
  %log_f3241.i = fadd <8 x float> %log_f3237.i, %log_f3240.i
  %log_f3242.i = fmul <8 x float> %log_f3241.i, %log_f3225.i
  %log_f3243.i = fmul <8 x float> splat (float 0xBF2BD01060000000), %27
  %log_f3244.i = fmul <8 x float> splat (float 5.000000e-01), %log_f3224.i
  %log_f3245.i = fadd <8 x float> %log_f3242.i, %log_f3243.i
  %28 = fsub <8 x float> %log_f3223.i, %log_f3244.i
  %log_f3246.i = fmul <8 x float> splat (float 0x3FE6300000000000), %27
  %log_f3247.i = fadd <8 x float> %28, %log_f3245.i
  %log_f3248.i = fadd <8 x float> %log_f3247.i, %log_f3246.i
  %log_f3250.i = and <8 x i32> %log_f324.i, splat (i32 -8388608)
  %29 = bitcast <8 x i32> %log_f3250.i to <8 x float>
  %log_f3252.i = and <8 x i32> %log_f327.i, splat (i32 2139095040)
  %30 = bitcast <8 x i32> %log_f3252.i to <8 x float>
  %log_f3255.i = or <8 x i32> %log_f3250.i, %log_f3252.i
  %log_f3256.i = bitcast <8 x i32> %log_f3255.i to <8 x float>
  %log_f3257.i = bitcast <8 x float> %log_f3248.i to <8 x i32>
  %log_f3259.i = or <8 x i32> %log_f3257.i, %log_f321.i
  %log_f3260.i = bitcast <8 x i32> %log_f3259.i to <8 x float>
  %log_f3263.i = or <8 x i32> %log_f324.i, %log_f327.i
  %log_f3264.i = bitcast <8 x i32> %log_f3263.i to <8 x float>
  %log_f3266.i = xor <8 x i32> %log_f3263.i, splat (i32 -1)
  %31 = bitcast <8 x i32> %log_f3266.i to <8 x float>
  %log_f3269.i = and <8 x i32> %log_f3266.i, %log_f3259.i
  %32 = bitcast <8 x i32> %log_f3269.i to <8 x float>
  %log_f3272.i = or <8 x i32> %log_f3255.i, %log_f3269.i
  %log_f3273.i = bitcast <8 x i32> %log_f3272.i to <8 x float>
  %33 = fmul <8 x float> %15, %15
  %34 = fmul <8 x float> %15, zeroinitializer
  %35 = fadd <8 x float> %34, splat (float 1.000000e+00)
  %36 = fmul <8 x float> %15, %35
  %37 = fadd <8 x float> %36, splat (float 0x402E2035A0000000)
  %38 = fmul <8 x float> %15, %37
  %39 = fadd <8 x float> %38, splat (float 0x4054C30B60000000)
  %40 = fmul <8 x float> %15, %39
  %41 = fadd <8 x float> %40, splat (float 0x406BB865A0000000)
  %42 = fmul <8 x float> %15, %41
  %43 = fadd <8 x float> %42, splat (float 0x4073519460000000)
  %44 = fmul <8 x float> %15, %43
  %45 = fadd <8 x float> %44, splat (float 0x406B0DB140000000)
  %46 = fmul <8 x float> %15, %45
  %47 = fadd <8 x float> %46, splat (float 0x404E0F3040000000)
  %48 = fadd <8 x float> %34, splat (float 0x3F07BC0960000000)
  %49 = fmul <8 x float> %15, %48
  %50 = fadd <8 x float> %49, splat (float 0x3FDFE818A0000000)
  %51 = fmul <8 x float> %15, %50
  %52 = fadd <8 x float> %51, splat (float 0x401A509F40000000)
  %53 = fmul <8 x float> %15, %52
  %54 = fadd <8 x float> %53, splat (float 0x403DE97380000000)
  %55 = fmul <8 x float> %15, %54
  %56 = fadd <8 x float> %55, splat (float 0x404E798EC0000000)
  %57 = fmul <8 x float> %15, %56
  %58 = fadd <8 x float> %57, splat (float 0x404C8E75A0000000)
  %59 = fmul <8 x float> %15, %58
  %60 = fadd <8 x float> %59, splat (float 0x40340A2020000000)
  %61 = fdiv <8 x float> %60, %47
  %62 = fmul <8 x float> %15, %33
  %63 = fmul <8 x float> %62, %61
  %64 = fmul <8 x float> %33, splat (float 5.000000e-01)
  %65 = fsub <8 x float> %63, %64
  %66 = fadd <8 x float> %15, %65
  %67 = tail call <8 x float> @llvm.fabs.v8f32(<8 x float> %15)
  %68 = fcmp olt <8 x float> %67, splat (float 0x3FDA8279A0000000)
  %69 = select <8 x i1> %68, <8 x float> %66, <8 x float> %log_f3273.i
  %70 = fneg <8 x float> %69
  %71 = fcmp ogt <8 x float> %69, splat (float -5.000000e+00)
  %72 = select <8 x i1> %71, <8 x float> splat (float 0x3FF805C5E0000000), <8 x float> splat (float 0x4006A9EFC0000000)
  %73 = select <8 x i1> %71, <8 x float> splat (float 0x3FCF91EC60000000), <8 x float> splat (float 0x3FF006DB60000000)
  %74 = select <8 x i1> %71, <8 x float> splat (float 0xBF711C9DE0000000), <8 x float> splat (float 0x3F8354AFC0000000)
  %75 = select <8 x i1> %71, <8 x float> splat (float 0xBF548A8100000000), <8 x float> splat (float 0xBF7F38BAE0000000)
  %76 = select <8 x i1> %71, <8 x float> splat (float 0x3F2CA65B60000000), <8 x float> splat (float 0x3F77824F60000000)
  %77 = select <8 x i1> %71, <8 x float> splat (float 0xBED26B5820000000), <8 x float> splat (float 0xBF6E17BCE0000000)
  %78 = select <8 x i1> %71, <8 x float> splat (float 0xBECD8E6AE0000000), <8 x float> splat (float 0x3F561B8E40000000)
  %79 = select <8 x i1> %71, <8 x float> splat (float 0x3E970966C0000000), <8 x float> splat (float 0x3F1A76AD60000000)
  %80 = select <8 x i1> %71, <8 x float> splat (float 0x3E5E2CB100000000), <8 x float> splat (float 0xBF2A3E1360000000)
  %81 = fsub <8 x float> splat (float -2.500000e+00), %69
  %82 = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %70)
  %83 = fadd <8 x float> %82, splat (float -3.000000e+00)
  %84 = select <8 x i1> %71, <8 x float> %81, <8 x float> %83
  %85 = fmul <8 x float> %80, %84
  %86 = fadd <8 x float> %79, %85
  %87 = fmul <8 x float> %84, %86
  %88 = fadd <8 x float> %78, %87
  %89 = fmul <8 x float> %84, %88
  %90 = fadd <8 x float> %77, %89
  %91 = fmul <8 x float> %84, %90
  %92 = fadd <8 x float> %76, %91
  %93 = fmul <8 x float> %84, %92
  %94 = fadd <8 x float> %75, %93
  %95 = fmul <8 x float> %84, %94
  %96 = fadd <8 x float> %74, %95
  %97 = fmul <8 x float> %84, %96
  %98 = fadd <8 x float> %73, %97
  %99 = fmul <8 x float> %84, %98
  %100 = fadd <8 x float> %72, %99
  %101 = select <8 x i1> %13, <8 x float> splat (float 0x7FF0000000000000), <8 x float> %100
  %102 = fmul <8 x float> %11, %101
  %103 = fmul <8 x float> %102, splat (float 0x3FF6A09E60000000)
  %104 = getelementptr inbounds [3 x [10000 x [64 x float]]], ptr %arg2, i64 0, i64 %broadcast_multiply_fusion.clone.invar_address.dim.0.06.us, i64 %broadcast_multiply_fusion.clone.invar_address.dim.1.04.us, i64 %index
  store <8 x float> %103, ptr %104, align 32, !alias.scope !4
  %index.next = add nuw i64 %index, 8
  %105 = icmp eq i64 %index.next, 64
  br i1 %105, label %middle.block, label %vector.body, !llvm.loop !7

middle.block:                                     ; preds = %vector.body
  %invar.inc5.us = add nuw nsw i64 %broadcast_multiply_fusion.clone.invar_address.dim.1.04.us, 1
  %exitcond8.not = icmp eq i64 %invar.inc5.us, %up_dim_1
  br i1 %exitcond8.not, label %broadcast_multiply_fusion.clone.loop_header.dim.1.broadcast_multiply_fusion.clone.loop_exit.dim.1_crit_edge.us, label %vector.ph, !llvm.loop !10

broadcast_multiply_fusion.clone.loop_header.dim.1.broadcast_multiply_fusion.clone.loop_exit.dim.1_crit_edge.us: ; preds = %middle.block
  %invar.inc.us = add nuw nsw i64 %broadcast_multiply_fusion.clone.invar_address.dim.0.06.us, 1
  %exitcond9.not = icmp eq i64 %invar.inc.us, %up_dim_0
  br i1 %exitcond9.not, label %return, label %broadcast_multiply_fusion.clone.loop_header.dim.1.preheader.us, !llvm.loop !12

return:                                           ; preds = %broadcast_multiply_fusion.clone.loop_header.dim.1.broadcast_multiply_fusion.clone.loop_exit.dim.1_crit_edge.us, %1
  ret ptr null
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.maximum.v8f32(<8 x float>, <8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.fabs.v8f32(<8 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #1

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 0}
!1 = !{}
!2 = !{i64 7680000}
!3 = !{i64 64}
!4 = !{!5}
!5 = !{!"result slice: {index:0, offset:0, size:7680000}", !6}
!6 = !{!"XLA host kernel broadcast_multiply_fusion.clone AA domain"}
!7 = distinct !{!7, !8, !9}
!8 = !{!"llvm.loop.isvectorized", i32 1}
!9 = !{!"llvm.loop.unroll.runtime.disable"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.unroll.disable"}
!12 = distinct !{!12, !11}
