
mod core;

use crate::core::engine::Value;
use crate::core::neural_network::MLP;


fn main() {

    let layerss  = MLP::new(3,&[4,4,1]);

    let xs = vec![
        vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
        vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
        vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
    ];
    
    let ys = vec![
        Value::new(1.0),
        Value::new(-1.0),
        Value::new(-1.0),
        Value::new(1.0),
    ];

    for epoch in 0..100 {
        
        let ys_pred: Vec<Value> = xs.iter()
            .map(|x| layerss.call(x.clone())[0].clone())
            .collect();
        
        // Compute loss
        let total_loss = ys.iter()
            .zip(ys_pred.iter())
            .map(|(y_true, y_pred)| (y_true.clone() - y_pred.clone()).pow(2.0))
            .fold(Value::new(0.0), |acc, loss| acc + loss);
        
        // Backward pass
        total_loss.backward();
        
        // Update weights
        let lr = 0.1;
        for param in layerss.parameters() {
            let new_data = param.data() - lr * param.grad();
            param.0.borrow_mut().data = new_data;
            param.set_grad(0.0);  // Important: zero gradients!
        }
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, total_loss.data());
        }
    }
 

    for (i, x) in xs.iter().enumerate() {
        let pred = layerss.call(x.clone())[0].data();
        let true_val = ys[i].data();
        println!("Sample {}: Pred = {:.4}, True = {:.4}", i, pred, true_val);
    }

} 

