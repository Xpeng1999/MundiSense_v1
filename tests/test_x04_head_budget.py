from emotioncore.x04.head_budget import project


def test_first_grid_projection_includes_every_arm_and_budget_stop():
    grid=[dict(train_seconds=10.,train_batches=100,validation_seconds=2.,validation_batches=20,
               elapsed_s=13.,epochs_executed=10,best_epoch=5) for _ in range(6)]
    counts=[dict(arm=a,grid_count=6 if a=='D0' else 18,selection_n=[1000]*5,
                 validation_n=[500]*5,refit_n=[1500]*6,outer_n=[600]*5)
            for a in ('D0','D1','D2','D3','D1_audioonly')]
    r=project(grid,counts,1.,.2,2.)
    assert r['passed']
    assert set(r['A3_arms'])=={'D0','D1','D2','D3','D1_audioonly'}
    assert r['selection_epochs_projection']==15
    assert r['refit_epochs_projection']==5
    assert r['total_projection_gpu_hours']>r['charged_gpu_hours']+r['A3_remaining_gpu_hours']
    assert r['total_projection_if_all_reach_50']>r['total_projection_gpu_hours']
    assert not project(grid,counts,19.9,.2)['passed']
