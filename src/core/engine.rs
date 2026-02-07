use std::{cell::RefCell, collections::HashSet, fmt::Display, ops::{Add, Div, Mul, Sub}, rc::Rc};



pub struct Value(pub Rc<RefCell<ValueData>>);

pub struct ValueData {
    pub data: f32,
    pub prev: Vec<Value>,
    pub backprop: Option<Box<dyn FnMut()>>,
    pub grad: f32,
    pub op: String,
}


// getter and setter core!!!
impl Value {
    pub fn data(&self) -> f32 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f32 {
        self.0.borrow().grad
    }

    pub fn set_grad(&self, g: f32) {
        self.0.borrow_mut().grad = g;
    }

    pub fn add_grad(&self, g: f32) {
        self.0.borrow_mut().grad += g;
    }

    pub fn op(&self) -> String {
        self.0.borrow_mut().op.clone()
    }

    pub fn prev(&self) -> Vec<Value> {
        self.0.borrow_mut().prev.clone()
    }
}

// forward pass core!!!
impl Value {
    pub fn new(val: f32) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            data: val,
            prev: Vec::new(),
            op: String::new(),
            grad: 0.0,
            backprop: None,
        })))
    }

    pub fn repr(&self) {
        println!("{}", self.0.borrow().data)
    }

    pub fn tanh(self) -> Value {
        let n = self.data();
        let t = ((2.0 * n).exp() - 1.0) / ((2.0 * n).exp() + 1.0);

        let out = Value::new(t);

        {
            let mut out_mut = out.0.borrow_mut();
            out_mut.op = "tanh".to_string();
            out_mut.prev.push(self.clone());
        }

        let out_cl  = out.clone();
        
         out.0.borrow_mut().backprop  = Some(Box::new(move|| {
            self.set_grad((1.0  - t * t) * out_cl.grad());
         })); 

        out
    }


    pub fn exp(self) -> Value {
        let num = self.data();
        let mut  output =  Value::new(num.exp());

        {
                output.0.borrow_mut().prev.push(self.clone());
                output.0.borrow_mut().op  = String::from("exp");       
        }

        let out_cl = output.clone();

        output.0.borrow_mut().backprop = Some(Box::new(move|| {
            self.0.borrow_mut().grad += out_cl.data() * out_cl.0.borrow().data;
        })); 
 
        output

    }
 
    pub fn pow(&self, exponent: f32) -> Value {
        let base_data = self.data();
        let output = Value::new(base_data.powf(exponent));
        
        {
            let mut output_mut = output.0.borrow_mut();
            output_mut.prev.push(self.clone());
            output_mut.op = format!("**{}", exponent);
        }

        let out_cl = output.clone();
        let base = self.clone();

        output.0.borrow_mut().backprop = Some(Box::new(move || {
            let out_grad = out_cl.0.borrow().grad;
            

            base.0.borrow_mut().grad += exponent * base_data.powf(exponent - 1.0) * out_grad;
        }));

        output
    }
       
}

// backprop core!!!
impl Value {
    
    pub fn backward(&self) {

        self.set_grad(1.00);

        let mut topo:Vec<Value> = Vec::new();
        let mut visited:HashSet<usize>  = std::collections::HashSet::new();

        self.build_topo(&mut topo,&mut visited);

        for node in topo.iter().rev() {
            let backprop_fn = {
                let mut borrowed = node.0.borrow_mut();
                borrowed.backprop.take() // Take ownership temporarily
            };
            
            if let Some(mut bp) = backprop_fn {
                bp();
            }
        }


    }


    fn build_topo(&self, topo: &mut Vec<Value>, visited: &mut HashSet<usize>) {

        let ptr = Rc::as_ptr(&self.0) as usize;
        

        if !visited.contains(&ptr) {
            visited.insert(ptr);
            

            for child in self.prev() {
                child.build_topo(topo, visited);
            }
            
            topo.push(self.clone());
        }
    }
}


// maths
impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {

        let out = Value::new(self.data() + rhs.data());

        {
            let mut out_mut = out.0.borrow_mut();
            out_mut.prev.push(self.clone());
            out_mut.prev.push(rhs.clone());
            out_mut.op = "+".to_string();
        }

        let a = self.clone();
        let b = rhs.clone();
        let out_clone = out.clone();

        out.0.borrow_mut().backprop = Some(Box::new(move || {
            let grad = out_clone.grad();
            a.add_grad(1.0 * grad);
            b.add_grad(1.0 * grad);
        }));

        out
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        let out = Value::new(self.data() * rhs.data());

        {
            let mut out_mut = out.0.borrow_mut();
            out_mut.prev.push(self.clone());
            out_mut.prev.push(rhs.clone());
            out_mut.op = "*".to_string();
        }

        let a = self.clone();
        let b = rhs.clone();
        let out_cl = out.clone();


        out.0.borrow_mut().backprop  = Some(Box::new(move|| {
            a.add_grad(b.data() * out_cl.grad() );
            b.add_grad(a.data() * out_cl.grad());
        }));

        out
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        let output = Value::new(self.data() / rhs.data());
        {
            output.0.borrow_mut().prev.push(self.clone());
            output.0.borrow_mut().prev.push(rhs.clone());

            output.0.borrow_mut().op =  String::from("/")
        }

        let out_cl = output.clone();
        let a =  self.clone();
        let b =  rhs.clone();

        output.0.borrow_mut().backprop  = Some(Box::new(move || {
            let out_grad  = out_cl.grad();
            a.0.borrow_mut().grad +=  (1 as f32).div(b.data()) * out_grad;
            b.0.borrow_mut().grad +=  -(a.data().div(b.data() * b.data())) * out_grad;
        }));


        output
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        let output = Value::new(self.data() - rhs.data());

       { 
        let mut  out  = output.0.borrow_mut();
        out.op = String::from("-");
        out.prev.push(self.clone());
        out.prev.push(rhs.clone());
        }

        let out_cl = output.clone();
        let a =  self.clone();
        let b =  rhs.clone();

        output.0.borrow_mut().backprop = Some(Box::new(move|| {
            let out_grad = out_cl.0.borrow().grad;

            a.0.borrow_mut().grad += 1.0 * out_grad;
            b.0.borrow_mut().grad += -1.0 * out_grad;
        }));

     output
    }
}


// etc

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let node = self.0.borrow();

        let prev_vals: Vec<f32> = node.prev
            .iter()
            .map(|p| p.0.borrow().data)
            .collect();

        write!(
            f,
            "Value(data={:.4}, prev={:?}, op={}, grad={:.4})",
            node.data, prev_vals, node.op, node.grad
        )
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        Value(self.0.clone())
    }
}

impl From<f32> for Value {
    fn from(val: f32) -> Self {
        Value::new(val)
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let node = self.0.borrow();
        f.debug_struct("Value")
            .field("data", &node.data)
            .field("grad", &node.grad)
            .field("op", &node.op)
            .finish()
    }
}

