use crate::core::engine::Value;
use rand::prelude::*;


#[derive(Debug)]
pub struct Neuron{
    pub weights: Vec<Value>,
    pub bias: Value,
}


impl Neuron {
    
    pub fn new(nin:usize) -> Self {

        let weights = (0..nin)  
        .map(|_| Value::new(rand::random_range(-1.0..=1.0)))
        .collect();
        let bias = Value::new(rand::random_range(-1.0..=1.0));
    
         Neuron { weights, bias }      

    }

    pub fn call(&self, x: &[Value]) -> Value {
        let mut sum = self.bias.clone();
        
        for (w, x_val) in self.weights.iter().zip(x.iter()) {
            sum = sum + (w.clone() * x_val.clone());
        }
        
        sum.tanh()  
    }


    pub fn _list(self,x:Vec<f32>) -> Vec<(f32,f32)>  {

        let out:Vec<(f32,f32)>  = self.weights.iter()
        .zip(x)
        .map(|(a,b)| (a.data(),b))
        .collect();

        out
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    } 
}



#[derive(Debug)]
pub struct Layer {
    pub neurons:Vec<Neuron>,
}


impl Layer {
    
    pub fn new(nin: usize, nout: usize) -> Self {
        let neurons = (0..nout)
            .map(|_| Neuron::new(nin))
            .collect();
        
        Layer { neurons }
    }

    pub fn call(&self, x: &[Value]) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.call(x))
            .collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }

}


// MLP

#[derive(Debug)]
pub struct MLP{
    pub layers:Vec<Layer>
}

impl MLP {
    
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let mut sz = vec![nin];
        sz.extend_from_slice(nouts);
        
        let layers = (0..nouts.len())
            .map(|i| Layer::new(sz[i], sz[i + 1]))
            .collect();
        
        MLP { layers }
    }
    
    pub fn call(&self, mut x: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            x = layer.call(&x);
        }
        x
    }


    pub fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}  