# Without sentiment
python -m luigi --module recommender_system.train TrainRS --sentiment-model without --recommender-model svd --local-scheduler
python -m luigi --module recommender_system.train TrainRS --sentiment-model without --recommender-model svdpp --local-scheduler
python -m luigi --module recommender_system.train TrainRS --sentiment-model without --recommender-model nmf --local-scheduler