import React from 'react'
import Header2 from './Header2'
import { Link } from 'react-router-dom'

function Footer() {
    return (
        <footer className='bg-dark text-center text-white mt-auto'>
            <div className='container p-4'>
                <section className='mb-4'>
                    <Link to="https://github.com/Messerschmidts101/Churn-Prediction-V1" className="btn btn-outline-light btn-floating m-1">
                        <i class="fa-brands fa-github"></i>
                    </Link>
                </section>
                <section className='mb-4'>
                    <p>
                        At TelCo Generic, we specialize in customer churn analysis, empowering businesses to reduce churn rates and enhance customer retention. Through advanced data analytics and predictive modeling, we uncover valuable insights and provide actionable strategies to optimize customer loyalty. Partner with us to unlock the potential of your customer base and drive sustainable growth through data-driven retention solutions.
                    </p>
                </section>
                <section className='container'>
                    <div className='row'>
                        <div className='col'>
                            <Header2>
                                WebDev Team
                            </Header2>
                            <p>Lintag, Juan Carlo</p>
                            <p>Olaguer, Matthew</p>
                            <p>Paano, Julius Angelo</p>
                        </div>
                        <div className='col'>
                            <Header2>
                                Data Science Team
                            </Header2>
                            <p>Alarcon, Chastin</p>
                            <p>David, Adrien Christian</p>
                            <p>Ferrer, Bryan Kristoffer</p>
                            <p>Señorin, Ereka Sheen</p>
                        </div>
                    </div>
                </section>
            </div>
        </footer>
    )
}

export default Footer